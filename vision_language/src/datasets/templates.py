"""
Text prompt templates for different dataset classes

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning

Modified from the codebase by Ilharco et al.,
at https://github.com/mlfoundations/task_vectors
"""

cars_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'i love my {c}!',
    lambda c: f'a photo of my dirty {c}.',
    lambda c: f'a photo of my clean {c}.',
    lambda c: f'a photo of my new {c}.',
    lambda c: f'a photo of my old {c}.',
]

cifar10_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'a low contrast photo of a {c}.',
    lambda c: f'a high contrast photo of a {c}.',
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a photo of a big {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a low contrast photo of the {c}.',
    lambda c: f'a high contrast photo of the {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the big {c}.',
]

cifar100_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'a low contrast photo of a {c}.',
    lambda c: f'a high contrast photo of a {c}.',
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a photo of a big {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a low contrast photo of the {c}.',
    lambda c: f'a high contrast photo of the {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the big {c}.',
]

dtd_template = [
    lambda c: f'a photo of a {c} texture.',
    lambda c: f'a photo of a {c} pattern.',
    lambda c: f'a photo of a {c} thing.',
    lambda c: f'a photo of a {c} object.',
    lambda c: f'a photo of the {c} texture.',
    lambda c: f'a photo of the {c} pattern.',
    lambda c: f'a photo of the {c} thing.',
    lambda c: f'a photo of the {c} object.',
]

eurosat_template = [
    lambda c: f'a centered satellite photo of {c}.',
    lambda c: f'a centered satellite photo of a {c}.',
    lambda c: f'a centered satellite photo of the {c}.',
]

food101_template = [
    lambda c: f'a photo of {c}, a type of food.',
]

gtsrb_template = [
    lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
    lambda c: f'a centered photo of a "{c}" traffic sign.',
    lambda c: f'a close up photo of a "{c}" traffic sign.',
]

mnist_template = [
    lambda c: f'a photo of the number: "{c}".',
]

imagenet_template = [
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
]

resisc45_template = [
    lambda c: f'satellite imagery of {c}.',
    lambda c: f'aerial imagery of {c}.',
    lambda c: f'satellite photo of {c}.',
    lambda c: f'aerial photo of {c}.',
    lambda c: f'satellite view of {c}.',
    lambda c: f'aerial view of {c}.',
    lambda c: f'satellite imagery of a {c}.',
    lambda c: f'aerial imagery of a {c}.',
    lambda c: f'satellite photo of a {c}.',
    lambda c: f'aerial photo of a {c}.',
    lambda c: f'satellite view of a {c}.',
    lambda c: f'aerial view of a {c}.',
    lambda c: f'satellite imagery of the {c}.',
    lambda c: f'aerial imagery of the {c}.',
    lambda c: f'satellite photo of the {c}.',
    lambda c: f'aerial photo of the {c}.',
    lambda c: f'satellite view of the {c}.',
    lambda c: f'aerial view of the {c}.',
]

stl10_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
]

sun397_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
]

svhn_template = [
    lambda c: f'a photo of the number: "{c}".',
]

caltech256_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a painting of a {c}.",
    lambda c: f"a plastic {c}.",
    lambda c: f"a sculpture of a {c}.",
    lambda c: f"a sketch of a {c}.",
    lambda c: f"a tattoo of a {c}.",
    lambda c: f"a toy {c}.",
    lambda c: f"a rendition of a {c}.",
    lambda c: f"a embroidered {c}.",
    lambda c: f"a cartoon {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"a plushie {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"art of a {c}.",
    lambda c: f"graffiti of a {c}.",
    lambda c: f"a drawing of a {c}.",
    lambda c: f"a doodle of a {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a painting of the {c}.",
    lambda c: f"the plastic {c}.",
    lambda c: f"a sculpture of the {c}.",
    lambda c: f"a sketch of the {c}.",
    lambda c: f"a tattoo of the {c}.",
    lambda c: f"the toy {c}.",
    lambda c: f"a rendition of the {c}.",
    lambda c: f"the embroidered {c}.",
    lambda c: f"the cartoon {c}.",
    lambda c: f"the {c} in a video game.",
    lambda c: f"the plushie {c}.",
    lambda c: f"the origami {c}.",
    lambda c: f"art of the {c}.",
    lambda c: f"graffiti of the {c}.",
    lambda c: f"a drawing of the {c}.",
    lambda c: f"a doodle of the {c}.",
]

fgvc_template = [
    lambda c: f"a photo of a {c}, a type of aircraft.",
    lambda c: f"a photo of the {c}, a type of aircraft.",
]

flowers102_template = [
    lambda c: f"a photo of a {c}, a type of flower."
]

pet_template = [
    lambda c: f"a photo of a {c}, a type of pet."
]

cub200_template = [
    lambda c: f"a photo of a {c}, a type of bird."
]

voc2007_template = [
    lambda c: f"a photo of a {c}."
]

country211_template = [
    lambda c: f"a photo i took in {c}.",
    lambda c: f"a photo i took while visiting {c}.",
    lambda c: f"a photo from my home country of {c}.",
    lambda c: f"a photo from my visit to {c}.",
    lambda c: f"a photo showing the country of {c}."
]

ucf101_template = [
    lambda c: f'a photo of a person {c}.',
    lambda c: f'a video of a person {c}.',
    lambda c: f'a example of a person {c}.',
    lambda c: f'a demonstration of a person {c}.',
    lambda c: f'a photo of the person {c}.',
    lambda c: f'a video of the person {c}.',
    lambda c: f'a example of the person {c}.',
    lambda c: f'a demonstration of the person {c}.',
    lambda c: f'a photo of a person using {c}.',
    lambda c: f'a video of a person using {c}.',
    lambda c: f'a example of a person using {c}.',
    lambda c: f'a demonstration of a person using {c}.',
    lambda c: f'a photo of the person using {c}.',
    lambda c: f'a video of the person using {c}.',
    lambda c: f'a example of the person using {c}.',
    lambda c: f'a demonstration of the person using {c}.',
    lambda c: f'a photo of a person doing {c}.',
    lambda c: f'a video of a person doing {c}.',
    lambda c: f'a example of a person doing {c}.',
    lambda c: f'a demonstration of a person doing {c}.',
    lambda c: f'a photo of the person doing {c}.',
    lambda c: f'a video of the person doing {c}.',
    lambda c: f'a example of the person doing {c}.',
    lambda c: f'a demonstration of the person doing {c}.',
    lambda c: f'a photo of a person during {c}.',
    lambda c: f'a video of a person during {c}.',
    lambda c: f'a example of a person during {c}.',
    lambda c: f'a demonstration of a person during {c}.',
    lambda c: f'a photo of the person during {c}.',
    lambda c: f'a video of the person during {c}.',
    lambda c: f'a example of the person during {c}.',
    lambda c: f'a demonstration of the person during {c}.',
    lambda c: f'a photo of a person performing {c}.',
    lambda c: f'a video of a person performing {c}.',
    lambda c: f'a example of a person performing {c}.',
    lambda c: f'a demonstration of a person performing {c}.',
    lambda c: f'a photo of the person performing {c}.',
    lambda c: f'a video of the person performing {c}.',
    lambda c: f'a example of the person performing {c}.',
    lambda c: f'a demonstration of the person performing {c}.',
    lambda c: f'a photo of a person practicing {c}.',
    lambda c: f'a video of a person practicing {c}.',
    lambda c: f'a example of a person practicing {c}.',
    lambda c: f'a demonstration of a person practicing {c}.',
    lambda c: f'a photo of the person practicing {c}.',
    lambda c: f'a video of the person practicing {c}.',
    lambda c: f'a example of the person practicing {c}.',
    lambda c: f'a demonstration of the person practicing {c}.',

]

caltech101_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a tattoo of the {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'the plushie {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a doodle of the {c}.',
]

mscoco_template = [
    lambda c: f'{c}'
]

dataset_to_template = {
    'Cars': cars_template,
    'CIFAR10': cifar10_template,
    'CIFAR100': cifar100_template,
    'DTD': dtd_template,
    'EuroSAT': eurosat_template,
    'Food101': food101_template,
    'GTSRB': gtsrb_template,
    'MNIST': mnist_template,
    'ImageNet': imagenet_template,
    'Webvision': imagenet_template,
    'RESISC45': resisc45_template,
    'STL10': stl10_template,
    'SUN397': sun397_template,
    'SVHN': svhn_template,
    'Caltech256': caltech256_template,
    'Caltech101': caltech101_template,
    'FGVCAircraft': fgvc_template,
    'Flowers102': flowers102_template,
    'OxfordIIITPet': pet_template,
    'CUB200': cub200_template,
    'PascalVOC': voc2007_template,
    'Country211': country211_template,
    'UCF101': ucf101_template,
    'mscoco': mscoco_template,
}


def get_templates(dataset_name):
    # Remove class-specific prefix
    if '_' in dataset_name:
        _, dataset_name = dataset_name.split('_')
        
    if dataset_name.endswith('Val'):
        return get_templates(dataset_name.replace('Val', ''))

    if "ImageNet" in dataset_name:
        return dataset_to_template["ImageNet"]
        
    assert dataset_name in dataset_to_template, f'Unsupported dataset: {dataset_name}'
    return dataset_to_template[dataset_name]
