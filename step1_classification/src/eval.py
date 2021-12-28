import os
import time
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class_dict = {}
train_class = {}


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_classes(classes_file_path):
    classes_file = open(classes_file_path, 'r')
    for line in classes_file.readlines():
        class_num, class_name = line.rstrip('\n').split('.')
        class_dict[class_name] = class_num


def load_tain_class():
    species_names = os.listdir(os.path.join(
        '..', '..', 'data', '8_class', 'train'))
    for i, specie_name in enumerate(species_names):
        train_class[i+1] = specie_name


class EvalDataset(Dataset):
    def __init__(self, image_folder, input_size, image_order_path=None):
        self.image_folder = image_folder
        self.input_size = input_size
        if image_order_path is None:
            self.images = os.listdir(image_folder)
        else:
            self.image_order_path = image_order_path
            # Load the eval image order from file
            self.images = [name.rstrip('\n') for name in open(
                self.image_order_path, 'r').readlines()]
        shape = (self.input_size, self.input_size)
        self.transforms = transforms.Compose([
            transforms.Resize(shape),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path)
        return self.transforms(image), image_name

    def __len__(self):
        return len(self.images)


class AugDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
        # self.trans = ['no', 'H', 'V', 'HV']
        self.trans = ['no', 'H']

        self.toPIL = transforms.ToPILImage()
        self.toTensor = transforms.ToTensor()
        self.toHFlip = transforms.RandomHorizontalFlip(p=1)
        self.toVFlip = transforms.RandomVerticalFlip(p=1)
        self.toHVFlip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1)
        ])

    def __getitem__(self, index):
        t = self.trans[index]
        image = torch.squeeze(self.inputs, dim=0)
        trans_image = self.toPIL(image)
        trans_image = self.transform(trans_image, t)
        trans_image = self.toTensor(trans_image)
        # print(trans_image.size())
        # trans_image = torch.unsqueeze(trans_image, dim=0)
        return trans_image

    def transform(self, image, t):
        if t == 'no':
            return image
        elif t == 'H':
            return self.toHFlip(image)
        elif t == 'V':
            return self.toVFlip(image)
        elif t == 'HV':
            return self.toHVFlip(image)
        else:
            raise 'Trans method error!'

    def __len__(self):
        return len(self.trans)


def load_model(model_path, model_name):
    input_size = 0

    # Select the input size of corresponding model
    if model_name == "resnet":
        """ Resnet50
        """
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        """
        input_size = 299

    elif model_name == 'swin_transformer_base_224':
        """ Swin Transformer with image size 224x224
        """
        input_size = 224

    elif model_name == 'swin_transformer_large_384':
        """ Swin Transformer with image size 384x384
        """
        input_size = 384

    else:
        print("Invalid model name, exiting...")
        exit()

    # Load the corresponding model weights
    model_ft = torch.load(model_path)
    return model_ft, input_size


def load_models(model_path_list, model_name):
    model_list = []
    target_size = None
    for model_path in model_path_list:
        model, input_size = load_model(model_path, model_name)
        if target_size is None:
            target_size = input_size
        else:
            if target_size != input_size:
                print(
                    f'Models sizes are different with {target_size} and {input_size}.')
                exit()
        model_list.append(model)
    return model_list, target_size


def models2eval(model_list):
    for i in range(len(model_list)):
        model_list[i].eval()
    return model_list


def models2device(model_list, device):
    for i in range(len(model_list)):
        model_list[i] = model_list[i].to(device)
    return model_list


def ensemble(inputs, model_list):
    result = None
    for model in model_list:
        if result is None:
            result = model(inputs)
        else:
            result = result + model(inputs)
    return result


def aug_ensemble(inputs, model_list, device):

    results = None
    aug_dataset = AugDataset(inputs)
    aug_dataloader = DataLoader(aug_dataset, batch_size=1,
                                pin_memory=True, num_workers=0)
    for trans_image in aug_dataloader:
        time.sleep(0.3)
        # print(trans_image.size())
        trans_image = trans_image.to(device)
        result = ensemble(trans_image, model_list)
        result = torch.softmax(result, 1)
        if results is None:
            results = result
        else:
            results += result
    return results


def eval_model(model_list, dataloader, result_path, device):
    # Open the file that store the eval result
    result = open(result_path, 'w')
    title = ['image'] + list(train_class.values())
    # print(title)
    title = ','.join(title)
    result.write(f'{title}\n')

    # Change the model mode to eval mode
    model_list = models2eval(model_list)

    # To eval each image according to eval image order
    for index, (inputs, inputs_name) in enumerate(tqdm(dataloader)):
        # Send input image to same device of model
        inputs = inputs.to(device)

        # eval the image
        # outputs = model(inputs)
        outputs = ensemble(inputs, model_list)
        # outputs = aug_ensemble(inputs, model_list, device)
        outputs = torch.softmax(outputs, 1)
        # print(outputs)
        # Take the hightest class number as eval result
        _, preds = torch.max(outputs, 1)

        # Convert the eval result number to final class name and class number
        name = train_class[int(preds[0])+1]
        # res = class_dict[name]

        # Record the eval reult to file
        if (index + 1) == len(dataloader):
            #     result.write(f'{inputs_name[0]} {name}')
            outputs = [f'{o:.4f}' for o in outputs[0].cpu().detach().numpy()]
            # print(outputs)
            prob = ','.join(outputs)
            # result.write(f'test_stg2/{inputs_name[0]},{prob}')
            result.write(f'{inputs_name[0]},{prob}')
        else:
            #     result.write(f'{inputs_name[0]} {name}\n')
            outputs = [f'{o:.4f}' for o in outputs[0].cpu().detach().numpy()]
            # print(outputs)
            prob = ','.join(outputs)
            # result.write(f'test_stg2/{inputs_name[0]},{prob}\n')
            result.write(f'{inputs_name[0]},{prob}\n')

    result.close()


def eval(stage=1):
    """ Load trained model
        Change the model name and timestamp to the model you want to eval.
    """
    model_name = 'swin_transformer_large_384'
    # timestamps = ['2021-12-20 12-40-50']
    # timestamps = ['2021-12-20 13-43-07']
    timestamps = ['2021-12-20 15-20-37']
    model_path_list = []
    for timestamp in timestamps:
        model_path = os.path.join(
            '..', 'model', timestamp+'-8class', f'{model_name}_{timestamp}.pkl')
        model_path_list.append(model_path)
    model_list, input_size = load_models(model_path_list, model_name)

    # Create evaluation dataset
    # eval_image_order_path = os.path.join('..', 'data', 'testing_img_order.txt')
    if stage == 1:
        eval_image_dir = os.path.join('..', '..', 'data', 'test_stg1')
    elif stage == 2:
        eval_image_dir = os.path.join('..', '..', 'data', 'test_stg2')
    else:
        raise "Stage Error!"
    eval_dataset = EvalDataset(
        eval_image_dir, input_size=input_size)

    # Create evaluation dataloader
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=1, pin_memory=True, num_workers=4)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    # model = model.to(device)
    model_list = models2device(model_list, device)

    """
        Load the classification classes names and numbers
            and store as dictionary, such as {class name: class number}.
    """
    # classes_file_path = os.path.join('..', 'data', 'classes.txt')
    # load_classes(classes_file_path)

    """
        Load train classes names and numbers and store as dictionary,
        such as {class number: class name}.
        This class number is different as above.
        This class number is according to the order of
            the class names in window directory order.
    """
    load_tain_class()

    # Create a timestamp for each eval result
    timestamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

    # The directory of the evaluation result stored
    result_dir = os.path.join(
        '..', 'result', f'eval_{model_name}_{timestamp}')
    # Create the above directory
    create_dir(result_dir)

    # the path of the evaluation result stored
    # result_path = os.path.join(
    # result_dir, f'eval_{model_name}_{timestamp}.txt')

    result_path = os.path.join(
        result_dir, f'eval_{model_name}_{timestamp}.csv')

    eval_model(model_list, eval_dataloader, result_path, device)
