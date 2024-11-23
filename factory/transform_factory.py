import torchaudio.transforms as T
import torch

class TransformFactory:
    @staticmethod
    def create(config):
        """
        Creates transforms for training, validation, and testing.

        Args:
            config: Configuration object.

        Returns:
            Transform callable for training, validation, and testing.
        """
        train_transforms = []

        # Add SpecAugment
        if config.augmentation.data_augmentation:
            if config.augmentation.augment_add_noise:
                train_transforms.append(RandomAddNoise(config=config))
            if config.augmentation.augment_spec_augment:
                train_transforms.append(SpecAugment(config=config))

        def compose(transforms):
            def composed(spectrogram):
                for transform in transforms:
                    spectrogram = transform(spectrogram)
                return spectrogram
            return composed

        return compose(train_transforms)


class RandomAddNoise:
    """Random noise addition to mel-spectrogram."""
    def __init__(self, config):
        self.noise_level = config.augmentation.noise_level
        

    def __call__(self, spectrogram):
        noise = torch.randn_like(spectrogram) * self.noise_level
        return spectrogram + noise

class SpecAugment:
    """SpecAugment: time and frequency masking."""
    def __init__(self, config, freq_mask_param=15, time_mask_param=35):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __call__(self, spectrogram):
        spectrogram = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)(spectrogram)
        spectrogram = T.TimeMasking(time_mask_param=self.time_mask_param)(spectrogram)
        return spectrogram