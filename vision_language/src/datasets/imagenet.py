import os
import torch
import torchvision
from .common import ImageFolderWithPaths, SubsetSampler
import numpy as np
from PIL import Image


imagenet_classnames = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
    "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
    "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
    "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
    "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
    "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
    "box turtle", "banded gecko", "green iguana", "Carolina anole",
    "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
    "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
    "American alligator", "triceratops", "worm snake", "ring-necked snake",
    "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
    "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
    "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
    "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
    "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
    "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
    "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
    "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
    "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
    "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
    "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
    "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
    "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
    "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
    "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
    "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
    "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
    "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
    "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
    "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
    "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
    "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
    "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
    "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
    "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
    "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
    "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
    "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
    "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
    "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
    "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
    "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
    "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
    "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
    "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
    "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
    "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
    "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
    "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
    "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
    "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
    "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
    "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
    "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
    "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
    "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
    "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
    "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
    "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
    "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
    "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
    "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
    "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
    "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
    "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
    "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
    "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
    "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
    "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
    "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
    "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
    "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
    "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
    "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
    "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
    "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
    "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
    "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
    "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
    "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
    "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
    "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
    "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
    "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
    "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
    "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
    "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
    "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
    "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
    "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
    "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
    "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
    "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
    "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
    "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
    "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
    "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
    "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
    "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
    "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
    "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
    "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
    "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
    "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
    "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
    "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
    "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
    "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
    "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
    "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
    "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
    "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
    "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
    "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
    "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
    "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
    "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
    "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
    "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
    "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
    "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
    "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
    "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
    "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
    "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
    "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
    "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
    "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
    "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
    "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
    "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
    "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
    "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
    "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
    "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
    "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
    "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
    "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
    "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
    "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
    "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
    "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
    "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
    "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
    "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
    "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
    "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
    "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
    "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
    "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
    "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
    "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
    "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
    "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
    "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
    "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
    "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
    "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
    "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"
]

class ImageNet:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=32):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = imagenet_classnames

        self.populate_train()
        self.populate_test()
    
    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), 'train')
        self.train_dataset = ImageFolderWithPaths(
            traindir,
            transform=self.preprocess)
        sampler = self.get_train_sampler()
        kwargs = {'shuffle' : True} if sampler is None else {}
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.get_test_sampler()
        )

    def get_test_path(self):
        test_path = os.path.join(self.location, self.name(), 'val_in_folder')
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, self.name(), 'val')
        return test_path

    def get_train_sampler(self):
        return None

    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def name(self):
        return 'imagenet/Data/CLS-LOC/'
    
    def class_ids(self):
        return list(range(1000))

class ImageNetTrain(ImageNet):

    def get_test_dataset(self):
        pass

class ImageNetK(ImageNet):

    def get_train_sampler(self):
        idxs = np.zeros(len(self.train_dataset.targets))
        target_array = np.array(self.train_dataset.targets)
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:self.k()] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetSampler(np.where(idxs)[0])
        return sampler


class ImageNetA(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classnames = [imagenet_classnames[i] for i in self.class_ids()]
    
    def populate_train(self):        
        self.train_loader = None

    def get_test_path(self):
        test_path = os.path.join(self.location, self.name())
        return test_path
    
    def name(self):
        return 'imagenet-a'

    def class_ids(self):
        #Class ids with regards to the original 1k of Imagenet
        return [6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107, 108, 110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307, 308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386, 397, 400, 401, 402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483, 486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614, 626, 627, 640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758, 763, 765, 768, 773, 774, 776, 779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870, 879, 880, 888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980, 981, 984, 986, 987, 988]

class ImageNetR(ImageNetA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classnames = [imagenet_classnames[i] for i in self.class_ids()]
        
    def name(self):
        return 'imagenet-r'

    def class_ids(self):
        #Class ids with regards to the original 1k of Imagenet
        return [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988] 

class ImageNetV2Freq(ImageNetA):
    def populate_test(self):
        #Because the folder names are string numbers .sort() is wrong and we cannot use ImageFolder or we get the wrong class order, class_ids(self) can also be used alternatively
        classes = os.listdir(os.path.join(self.location, self.name()))
        classes = [int(c) for c in classes]
        classes.sort()
        paths = []
        targets = []
        for c in classes:
            files = [os.path.join(self.location, self.name(), str(c), f) for f in os.listdir(os.path.join(self.location, self.name(), str(c)))]
            paths += files
            targets += [c] * len(files)
                
        self.test_dataset = BasicImageDataset(paths, targets, self.preprocess)
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.get_test_sampler()
        )

    def name(self):
        return 'imagenetv2-matched-frequency/'

    def class_ids(self):
        return list(range(1000))
    
class ImageNetV2Thresh(ImageNetV2Freq):
    def name(self):
        return 'imagenetv2-threshold0.7/'

class ImageNetV2Top(ImageNetV2Freq):
    def name(self):
        return 'imagenetv2-top-images/'

class ImageNetSketch(ImageNetA):
    def name(self):
        return 'ImageNet-Sketch/sketch'

    def class_ids(self):
        return list(range(1000))

    
class BasicImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, targets, transform=None):
        self.paths = paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        with open(self.paths[idx], "rb") as f:
            img = Image.open(f)
            image = img.convert("RGB")
        label = self.targets[idx]
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


class Webvision(ImageNet):
    NAME="webvision"
    def populate_train(self):
        with open(os.path.join(self.location, f'{self.NAME}/info/train_filelist_google.txt')) as f:
            lines=f.readlines()
            train_imgs = []
            targets = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                train_imgs.append(os.path.join(self.location, f'{self.NAME}', img))
                targets.append(target)
                
        self.train_dataset = BasicImageDataset(train_imgs, targets, self.preprocess)
        sampler = self.get_train_sampler()
        kwargs = {'shuffle' : True} if sampler is None else {}
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )
        
            
class WebvisionVal(Webvision):
    def populate_test(self):
        with open(os.path.join(self.location, f'{self.NAME}/info/val_filelist.txt')) as f:
            lines=f.readlines()
            val_imgs = []
            targets = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                val_imgs.append(os.path.join(self.location, f'{self.NAME}/val_images_256', img))
                targets.append(target)
                
        self.test_dataset = BasicImageDataset(val_imgs, targets, self.preprocess)
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.get_test_sampler()
        )
