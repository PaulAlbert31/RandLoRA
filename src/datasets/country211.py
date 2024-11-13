"""Country211 data utilities

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning
"""

import os
import torch
from pathlib import Path
from typing import Callable, Optional

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg

class Country211Base:
    def __init__(self,
                 preprocess,
                 test_split,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        self.train_dataset = PyTorchCountry211(location, 'train', transform=preprocess, download=True)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = PyTorchCountry211(location, test_split, transform=preprocess, download=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.classnames = [
            "Andorra",
            "United Arab Emirates",
            "Afghanistan",
            "Antigua and Barbuda",
            "Anguilla",
            "Albania",
            "Armenia",
            "Angola",
            "Antarctica",
            "Argentina",
            "Austria",
            "Australia",
            "Aruba",
            "Aland Islands",
            "Azerbaijan",
            "Bosnia and Herzegovina",
            "Barbados",
            "Bangladesh",
            "Belgium",
            "Burkina Faso",
            "Bulgaria",
            "Bahrain",
            "Benin",
            "Bermuda",
            "Brunei Darussalam",
            "Bolivia",
            "Bonaire, Saint Eustatius and Saba",
            "Brazil",
            "Bahamas",
            "Bhutan",
            "Botswana",
            "Belarus",
            "Belize",
            "Canada",
            "DR Congo",
            "Central African Republic",
            "Switzerland",
            "Cote d'Ivoire",
            "Cook Islands",
            "Chile",
            "Cameroon",
            "China",
            "Colombia",
            "Costa Rica",
            "Cuba",
            "Cabo Verde",
            "Curacao",
            "Cyprus",
            "Czech Republic",
            "Germany",
            "Denmark",
            "Dominica",
            "Dominican Republic",
            "Algeria",
            "Ecuador",
            "Estonia",
            "Egypt",
            "Spain",
            "Ethiopia",
            "Finland",
            "Fiji",
            "Falkland Islands",
            "Faeroe Islands",
            "France",
            "Gabon",
            "United Kingdom",
            "Grenada",
            "Georgia",
            "French Guiana",
            "Guernsey",
            "Ghana",
            "Gibraltar",
            "Greenland",
            "Gambia",
            "Guadeloupe",
            "Greece",
            "South Georgia and South Sandwich Is.",
            "Guatemala",
            "Guam",
            "Guyana",
            "Hong Kong",
            "Honduras",
            "Croatia",
            "Haiti",
            "Hungary",
            "Indonesia",
            "Ireland",
            "Israel",
            "Isle of Man",
            "India",
            "Iraq",
            "Iran",
            "Iceland",
            "Italy",
            "Jersey",
            "Jamaica",
            "Jordan",
            "Japan",
            "Kenya",
            "Kyrgyz Republic",
            "Cambodia",
            "St. Kitts and Nevis",
            "North Korea",
            "South Korea",
            "Kuwait",
            "Cayman Islands",
            "Kazakhstan",
            "Laos",
            "Lebanon",
            "St. Lucia",
            "Liechtenstein",
            "Sri Lanka",
            "Liberia",
            "Lithuania",
            "Luxembourg",
            "Latvia",
            "Libya",
            "Morocco",
            "Monaco",
            "Moldova",
            "Montenegro",
            "Saint-Martin",
            "Madagascar",
            "Macedonia",
            "Mali",
            "Myanmar",
            "Mongolia",
            "Macau",
            "Martinique",
            "Mauritania",
            "Malta",
            "Mauritius",
            "Maldives",
            "Malawi",
            "Mexico",
            "Malaysia",
            "Mozambique",
            "Namibia",
            "New Caledonia",
            "Nigeria",
            "Nicaragua",
            "Netherlands",
            "Norway",
            "Nepal",
            "New Zealand",
            "Oman",
            "Panama",
            "Peru",
            "French Polynesia",
            "Papua New Guinea",
            "Philippines",
            "Pakistan",
            "Poland",
            "Puerto Rico",
            "Palestine",
            "Portugal",
            "Palau",
            "Paraguay",
            "Qatar",
            "Reunion",
            "Romania",
            "Serbia",
            "Russia",
            "Rwanda",
            "Saudi Arabia",
            "Solomon Islands",
            "Seychelles",
            "Sudan",
            "Sweden",
            "Singapore",
            "St. Helena",
            "Slovenia",
            "Svalbard and Jan Mayen Islands",
            "Slovakia",
            "Sierra Leone",
            "San Marino",
            "Senegal",
            "Somalia",
            "South Sudan",
            "El Salvador",
            "Sint Maarten",
            "Syria",
            "Eswatini",
            "Togo",
            "Thailand",
            "Tajikistan",
            "Timor-Leste",
            "Turkmenistan",
            "Tunisia",
            "Tonga",
            "Turkey",
            "Trinidad and Tobago",
            "Taiwan",
            "Tanzania",
            "Ukraine",
            "Uganda",
            "United States",
            "Uruguay",
            "Uzbekistan",
            "Vatican",
            "Venezuela",
            "British Virgin Islands",
            "United States Virgin Islands",
            "Vietnam",
            "Vanuatu",
            "Samoa",
            "Kosovo",
            "Yemen",
            "South Africa",
            "Zambia",
            "Zimbabwe"
        ]

class Country211(Country211Base):
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        super().__init__(preprocess, "test", location, batch_size, num_workers)

class Country211Val(Country211Base):
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        super().__init__(preprocess, "valid", location, batch_size, num_workers)

class PyTorchCountry211(ImageFolder):
    """`The Country211 Data Set <https://github.com/openai/CLIP/blob/main/data/country211.md>`_ from OpenAI.

    This dataset was built by filtering the images from the YFCC100m dataset
    that have GPS coordinate corresponding to a ISO-3166 country code. The
    dataset is balanced by sampling 150 train images, 50 validation images, and
    100 test images images for each country.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"valid"`` and ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/country211/``. If dataset is already downloaded, it is not downloaded again.
    """

    _URL = "https://openaipublic.azureedge.net/clip/data/country211.tgz"
    _MD5 = "84988d7644798601126c29e9877aab6a"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self._split = verify_str_arg(split, "split", ("train", "valid", "test"))

        root = Path(root).expanduser()
        self.root = str(root)
        self._base_folder = root / "country211"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        super().__init__(str(self._base_folder / self._split), transform=transform, target_transform=target_transform)
        self.root = str(root)

    def _check_exists(self) -> bool:
        return self._base_folder.exists() and self._base_folder.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)

        
    def __getitem__(self, index: int):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            # return index, sample, target, path
            return sample, target
