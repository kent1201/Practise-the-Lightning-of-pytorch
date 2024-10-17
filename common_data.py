import albumentations as A


# LABELS_LIST = ['CP00', 'CP03', 'CP08', 'CP09', 'DR02', 
#                'IT03', 'IT07', 'IT08', 'IT09',
#                'PASSCP06', 'PASSDIRTY', 'PASSOTHER', 'PASSOXIDATION', 'PASSSCRATCHES', 'SHORTCP06', 'SHORTOTHER']

SRC_DIR = r'/workspace/Datasets/animals_64classes/all'
DATASET_DIR = r'/workspace/Datasets/animals_64classes/Ver_001_20241015'
# REFERENCE_PATH = r"D:\datasets\K2_datasets\CIMS_230907\train\CP00"

PROJECT = {
                "K2": {
                        "labels": ['CP00', 'CP03', 'CP08', 'CP09', 'DR02', 'IT03', 'IT07', 'IT08', 'IT09',
                                        'PASSCP06', 'PASSDIRTY', 'PASSOTHER', 'PASSOXIDATION', 'PASSSCRATCHES', 'SHORTCP06', 'SHORTOTHER'],
                },
                "animals_64classes": {
                        "labels": ["antelope", "bee", "buffalo", "cat", "chinchilla", "crocodile", "dolphin", "eagle", "ferret", "frog",
                                "goose", "hawk", "hyena", "kangaroo", "leopard", "mole", "otter", "peacock", "raccoon", "snail",
                                "squid", "wolf", "bear", "bison", "butterfly", "cheetah", "cow", "deer", "donkey", "elephant", "flamingo",
                                "giraffe", "gorilla", "hedgehog", "iguana", "koala", "lizard", "mongoose", "owl", "penguin", "seal",
                                "snake", "walrus", "beaver", "blackbird", "camel", "chimpanzee", "crab", "dog", "duck", "falcon", "fox",
                                "goat", "grasshopper", "hippopotamus", "jaguar", "lemur", "lynx", "ostrich", "panda", "porcupine", "sheep",
                                "spider", "whale"],
                }
        }

# transform = A.Compose([ 
#                     A.HorizontalFlip(p=0.5),
#                     A.VerticalFlip(p=0.5),
#                     A.Perspective(p=0.5),
#                     A.Transpose(p=0.3),
#                     A.RandomGridShuffle(p=0.3),
#                     A.OneOf([A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.4),
#                             A.Affine(p=0.4)]),
#                 ])