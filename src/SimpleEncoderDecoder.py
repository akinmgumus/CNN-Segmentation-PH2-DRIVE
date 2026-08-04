import torch.nn as nn

class SimpleEncoderDecoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1): # We assume binary segmentation because of DRIVE and PH2 datasets have lesions/vessels masks or not
        # and this model would be used for both datasets
        super(SimpleEncoderDecoder, self).__init__()    # Super is for inheriting nn.Module. If we don't call it, nn.Module's __init__ method won't be called.

        # Input size: batch_size x in_channels x H x W

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Image size H/2 x W/2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Image size H/4 x W/4
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # Image size H/2 x W/2
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # Image size H x W
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, out_channels, kernel_size=1)            # Final output layer
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x    # Logits
    

