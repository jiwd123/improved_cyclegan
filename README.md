# model 0-7
log download：链接：https://pan.baidu.com/s/1V3zdLIuSrBpUU9PY01eslA?pwd=cjq4 
提取码：cjq4 
# AT_model
log download：链接：https://pan.baidu.com/s/1valo208Q3HbrPSZToc05Lw?pwd=q0cc 
提取码：q0cc 
# Predict
```python
python test --cuda
```
# Post-process
Threshold selection post-processing strategy based on the RGB histogram
```python
python cthcl.py
```
Spatial Coordinate-based Target Background Separation Method
```python
python hcl.py
```
# Generate Dataset
1. Prepare soil images
2. Ensure operation Predict and Post-process
3.
```python
python touming.py
```
