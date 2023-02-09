import mindspore
from addict import Dict
from .det_dataset import DetDataset
#from .rec_dataset import RecLMDBDataset, RecTextDataset


support_dataset_types = ['DetDataset', 'RecDataset']


def build_dataset(dataset_config: dict, transform_config: dict, loader_config: dict, is_train=True):
    '''
    Args:
        dataset_config (dict): config dict for the dataset, keys:
            name:
            image_dir: directory of data, 
                image folder path for detection task
                lmdb folder path for lmdb dataset
                co
            annot_file (optional for recognition): annotation file path

        transform_config (dict): config dict for image and label transformation
            
    Return:
        data loader, for feeding data batch to network

    Examples:
         
        .dataset_name/
        ├── train/
        │  ├── class1/
        │  │   ├── 000001.jpg
        │  │   ├── 000002.jpg
        │  │   └── ....
        │  └── class2/
        │      ├── 000001.jpg
        │      ├── 000002.jpg
        │      └── ....
        └── eval/
         
    
    '''
    # parse config 
    module_name = dataset_config['name']

    assert module_name in support_dataset_types, "Invalid dataset name"

    dataset_class = eval(module_name)
    
    # transform op list for image and annotation
    transform_list = create_transforms(transform_config)

    
    dataset_class(image_dir, annot_file, )

    
    

    # parse files 

    # create transforms

    # create dataset iterator
    
    # create batch loader


    
    

if __name__ == "__main__":
    # det 
    dataset_config = {
            'name': 'DetDataset',
            'data_dir': ' /data/ocr_datasets/ic15/text_localization/train/ch4_training_images',
            'label_file_list': ['/data/ocr_datasets/ic15/text_localization/train/train_icdar15_label.txt'] 
            #'ratio_list': [1.0]
            }
    # transform_op_name -  param values 
    transform_config = {
            
            }

    loader_config = {
            'shffule': True, # TODO: tbc
            'batch_size': 8,
            'drop_remainder': True,
            'num_workers': 8
            }
    
    # rec
    
