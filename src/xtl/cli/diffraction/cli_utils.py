from enum import Enum

from xtl.diffraction.images.images import Image


class IntegrationErrorModel(Enum):
    NONE = 'None'
    POISSON = 'poisson'
    VARIANCE = 'variance'

class IntegrationRadialUnits(Enum):
    TWOTHETA_DEG = '2th_deg'
    Q_NM = 'q_nm'


def get_image_frames(images: list[str]) -> list[Image]:
    opened_images = []
    for i, img in enumerate(images):
        parts = img.split(':')
        if len(parts) == 1:
            file = parts[0]
            frame = 0
        elif len(parts) == 2:
            file = parts[0]
            if parts[1].isnumeric():
                frame = int(parts[1])
            else:
                raise ValueError(f'Invalid frame index for image [{i}]: {parts[1]!r}')
        else:
            raise ValueError(f'Invalid image format for image [{i}]: {img!r}')

        image = Image()
        try:
            image.open(file=file, frame=frame, is_eager=False)
        except Exception as e:
            raise ValueError(f'Failed to open image [{i}]: {img!r}') from e
        opened_images.append(image)

    return opened_images
