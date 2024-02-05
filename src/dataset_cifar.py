# dataloader for MNIST, CIFAR, TMED, and private AS datasets
from PIL import Image
import torchvision

def index_to_quadrant_id(index):
    # works w/ integers and numpy arrays
    img_index = index // 4
    quadrant_id = index % 4
    return img_index, quadrant_id

# function to extract quadrant from PIL image
def get_quadrant(img, quadrant_id):
    # PIL image -> PIL image
    w, h = img.size
    match quadrant_id:
    # quadrant ID is 0:top_left, 1:top_right, 2:bottom_left, 3:bottom_right
        case 0:
            window = (0, 0, w//2, h//2)
        case 1:
            window = (w//2, 0, w, h//2)
        case 2:
            window = (0, h//2, w//2, h)
        case 3:
            window = (w//2, h//2, w, h)
        case _:
            print("Invalid quadrant ID")
    img = img.crop(window)
    img = img.resize((w, h))
    return img

class CIFAR10_Quadrants(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.init_pseudolabels()
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Accesses a quadrant of an image in CIFAR
        Args:
            index (int): quadrant-based index, which is equal to 4*image_index + quadrant ID
            quadrant ID is 0:top_left, 1:top_right, 2:bottom_left, 3:bottom_right
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_index, quadrant_id = index_to_quadrant_id(index)
        img = self.data[img_index]
        #target = int(self.targets[img_index])
        target = self.pseudo[img_index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = get_quadrant(img, quadrant_id)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data) * 4
        
    def init_pseudo(self):
        len_data = len(self.data) * 4
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.pseudo = np.zeros(len_data, len(self.classes)))
        for i in range(len_data):
            img_index, _ = index_to_quadrant_id(index)
            label = int(self.targets[img_index])
            self.pseudo[i][label] = 1.0
        
    def get_pseudo(self):
        return self.pseudolabels
        
    def set_pseudo(self, new_pseudo):
        assert new_pseudo.shape == self.pseudo.shape
        self.pseudo = new_pseudo