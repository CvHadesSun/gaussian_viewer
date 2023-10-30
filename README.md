a simple viewer to view gaussian-splatting model from free viewpoints.

<video width="640" height="480" controls>
  <sourc src="assets/legao.mp4" type="video/mp4">
</video>

<video width="640" height="480" controls>
  <sourc src="assets/bj-test.mp4" type="video/mp4">
</video>

# install 

test with cuda 11.3 and torch 1.12.1

```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip install opencv-python 
pip install dearpygui

pip install engine/submodules/diff-gaussian-rasterization
pip install engine/submodules/simple-knn

```

# run

```shell
python demo.py
```

# reference
- [toch-ngp](https://github.com/ashawkey/torch-ngp)
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
