class GlobalPath:
    @property
    def color_event_dataset(self):
        raise NotImplementedError("Please define color event dataset root.")


class DESKTOP(GlobalPath):
    @property
    def color_event_dataset(self):
        return "D:\\Dataset\\3-ColorEventData\\Dataset\\"

    @property
    def color_event_bags(self):
        return "D:\\Dataset\\3-ColorEventData\\BAG\\"


class VLGPU8x87(GlobalPath):
    @property
    def color_event_dataset(self):
        return "./dataset/1-color-event-dataset/Dataset/"

    @property
    def alpx_event_dataset(self):
        return "./dataset/01-EG-VSR-1023-2022/"

    @property
    def alpx_vsr_dataset(self):
        return "./dataset/01-EG-VSR-1023-2022/01-2022-10-24-vsr/"


global_path = VLGPU8x87()
