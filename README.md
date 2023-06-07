# Bicubic++: Slim, Slimmer, Slimmest - Designing an Industry-Grade Super-Resolution Network (:rocket: Winner of NTIRE @ CVPR 2023 RTSR Challange Track 2)

**Bahri Batuhan Bilecen, Mustafa Ayazoglu**

 [\[Preprint\]](https://arxiv.org/abs/2305.02126)
 [\[CVF Open Access\]](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Bilecen_Bicubic_Slim_Slimmer_Slimmest_-_Designing_an_Industry-Grade_Super-Resolution_Network_CVPRW_2023_paper.html)
 
**Abstract:** We propose a real-time and lightweight single-image super-resolution (SR) network named Bicubic++. Despite using spatial dimensions of the input image across the whole network, Bicubic++ first learns quick reversible downgraded and lower resolution features of the image in order to decrease the number of computations. We also construct a training pipeline, where we apply an end-to-end global structured pruning of convolutional layers without using metrics like magnitude and gradient norms, and focus on optimizing the pruned network's PSNR on the validation set. Furthermore, we have experimentally shown that the bias terms take considerable amount of the runtime while increasing PSNR marginally, hence we have also applied bias removal to the convolutional layers. 

Our method adds ~1dB on Bicubic upscaling PSNR for all tested SR datasets and runs with ~1.17ms on RTX3090 and ~2.9ms on RTX3070, for 720p inputs and 4K outputs, both in FP16 precision. *Bicubic++ won NTIRE 2023 RTSR Track 2 x3 SR competition and is the fastest among all competitive methods.* Being almost as fast as the standard Bicubic upsampling method, we believe that Bicubic++ can set a new industry standard.


![teaser](/figures/teaser.png)

## Installation 
1. Clone the repository.
       
       git clone https://github.com/aselsan-research-imaging-team/bicubic-plusplus.git
2. Install the dependencies. We recommend using a virtual environment to manage the packages.
    * Python 3.7
    * PyTorch 1.7.1
    * CUDA 11.0
    * Other packages (numpy, opencv-python, pytorch-lightning, pyyaml)
      
## Test
1. Modify validation dataset paths (`data.val.lr_path` and `data.val.hr_path`) in `configs/conf.yaml` accordingly. Make sure that `load_pretrained` and `pretrained_path` are set correctly. You may change `loader.val.batch_size` to speed up the inference.
2. Run test code.
       
       python test.py
## Train
The proposed three-stage training pipeline code will be published soon.

## Citation

    @InProceedings{Bilecen_2023_CVPR,
        author    = {Bilecen, Bahri Batuhan and Ayazoglu, Mustafa},
        title     = {Bicubic++: Slim, Slimmer, Slimmest - Designing an Industry-Grade Super-Resolution Network},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        month     = {June},
        year      = {2023},
        pages     = {1623-1632}
    }

## License and Acknowledgement

This work is under CC BY-NC-SA 4.0 license.
