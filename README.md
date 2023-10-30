a simple viewer to view gaussian-splatting model from free viewpoints.

![legao](assets/legao.gif)


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
