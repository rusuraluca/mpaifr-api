import re
from os.path import basename
from torchvision.datasets import ImageFolder


age_cutoffs = [25, 55]
genders = [0, 1]
folder_pattern = r'_|\.'

class ImageLoader(ImageFolder):
    def __init__(self,
                root,
                pattern=folder_pattern,
                position_age=1,
                position_gender=1,
                cutoffs=age_cutoffs,
                transform=None):
        super().__init__(root, transform=transform)
        self.pattern = pattern
        self.position_age = position_age
        self.position_gender = position_gender
        self.cutoffs = cutoffs

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        age = self.path2age(path, self.pattern, self.position_age)
        age_group = self.find_age_group(age)
        gender = self.path2gender(path, self.pattern, self.position_gender)
        gender_group = self.find_gender_group(gender)
        return img, label, age_group, gender_group

    @staticmethod
    def path2age(path, pat, pos):
        return int(re.split(pat, basename(path))[pos])

    @staticmethod
    def path2gender(path, pat, pos):
        components = path.split('/')
        name_gender_dir = components[-2]
        gender_str = re.split(pat, name_gender_dir)[pos]
        return 0 if gender_str.lower().startswith('m') else 1

    def find_age_group(self, age):
        age_group = next((i for i, cutoff in enumerate(self.cutoffs) if age <= cutoff), len(self.cutoffs))
        return age_group

    @staticmethod
    def find_gender_group(gender):
        gender_group = gender
        return gender_group
