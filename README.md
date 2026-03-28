
<!-- PROJECT LOGO -->
<h1 align="center" style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap;">
  <img src="./media/ec3rlogo.png" alt="logo" width="60" style="margin-right: 10px;">
  EC3R-SLAM: Efficient and Consistent Monocular Dense SLAM with Feed-Forward 3D Reconstruction
</h1>

    <strong> Fabien Bonard<sup>1</sup></strong></a>
    ·
<strong> Raymond Ghandour<sup>2</sup></strong></a>
  </p>
  <p align="center">
      <strong><sup>1 </sup>IBISC Lab, Université Paris-Saclay, France,  <sup>2 </sup> American University of the Middle East, Kuwait, 
      <strong><h4 align="center"><a href="https://arxiv.org/html/2510.02080v1" target="_blank">Paper</a> | <a href="https://h0xg.github.io/ec3r/" target="_blank">Project Website</a></h4></strong>
  </strong></p>



## Tracking
<p align="center">
    <img src="./media/tracking.gif" alt="rerun_eg" width="100%">
</p>

## Mapping
<p align="center">
    <img src="./media/mapping.gif" alt="rerun_eg" width="100%">
</p>

## Running

To run the system, first install the required dependencies:
```bash
pip install -r requirements.txt
```
Then launch the pipeline with:

```bash
python3 main.py \
    --sourcepath <path_to_dataset> \
    --config <path_to_config>
```
## Citation

If you find our code or paper useful, please cite
```bibtex
@article{hu2025ec3r,
  title={EC3R-SLAM: Efficient and Consistent Monocular Dense SLAM with Feed-Forward 3D Reconstruction},
  author={Hu, Lingxiang and Oufroukh, Naima Ait and Bonardi, Fabien and Ghandour, Raymond},
  journal={arXiv preprint arXiv:2510.02080},
  year={2025}
}
```
## Contact
Contact [Lingxiang Hu](mailto:hulxhlx@gmail.com) for questions, comments and reporting bugs.
