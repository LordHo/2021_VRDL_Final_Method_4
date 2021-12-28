import timm
from pprint import pprint
import os

# model_list = open('model list.txt', 'w')
# model_names = timm.list_models(pretrained=True)
# print('\n'.join(model_names), file=model_list)
# model_list.close()

f = open(os.path.join('model_structure',
         'swin_large_patch4_window12_384_in22k.txt'), 'w')
model = timm.create_model(
    'swin_large_patch4_window12_384_in22k', pretrained=True)
print(model, file=f)
f.close()
