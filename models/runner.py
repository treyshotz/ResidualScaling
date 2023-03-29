# Test the original resnet 50
import torch

from models.TrialPipeline import TrialPipeline


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    original_resnet = TrialPipeline(batch_size=128,
                                    num_workers=1,
                                    epochs=30,
                                    lr=0.001,
                                    device=device,
                                    scaled=False
                                    )

    # original_resnet.train_pipeline()


    # Compare to original resnet 50
    scaling_factors = 1
    original_resnet = TrialPipeline(batch_size=128,
                                    num_workers=1,
                                    epochs=30,
                                    lr=0.001,
                                    device=device,
                                    scaled=True
                                    )

    original_resnet.train_pipeline()
