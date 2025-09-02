import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="mit_b1",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,
    activation=None
)
