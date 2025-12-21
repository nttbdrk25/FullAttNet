# Full-plane Attentive Patterns Adopted by an Adaptative Lightweight Backbone for Image Recognition

**Abstract:**
A novel attention mechanism is proposed by considering the full planes of a given tensor so that the learning
process can concentrate on the complete aspects of a concerning
object. Accordingly, we split T into three plane-based tensors
corresponding to its three orthogonal planes. For each planebased tensor, its multiple statistical features will be exploited to
enrich the channel-spatial information for attentive patterns that
are integrated for a robust full-plane attention (named FullAtt).
Due to the orthogonal computations, FullAtt would not adapt
with the normal backbone of all existing lightweight networks,
whose attentive process has been formed to only focus on one
side of T . To this end, an adaptive network (called FullAttNet)
is proposed by addressing the diversity of stride-based features
(MuS) that have been extracted by multiple stride-based FullAtt
blocks (MuSFullAtt) for further discriminative description. To
the best of our knowledge, it is the first time that these stridebased patterns of T have been exploited within a perceptive block
to capture diverse spatial information for the learning process.
Experiments for image recognition on benchmark datasets have
validated the superiority of our proposals.

<u>**Training and validating FullAttNet on benchmark datasets:**</u>

- For training FullAttNet on datasets Stanford Dogs and challenging subset ReIN100 of ImageNet:
```
$ python Train_FullAttNet_StanfordDogs.py
$ python Train_FullAttNet_ReIN.py
```
- For validating FullAttNet on datasets Stanford Dogs and challenging subset ReIN100 of ImageNet::
```
$ python Train_FullAttNet_StanfordDogs.py --evaluate
$ python Train_FullAttNet_ReIN.py --evaluate
```
<u>**Note:**</u>
- Subject to your system, modify these files (*.py) to have the right path to dataset

- For the instance of validation of FullAttNet on challenging subset ReIN100 of ImageNet, download its trained model at: [Click here](https://drive.google.com/drive/folders/1l0dPxui5KF5HNI87YUX_HRebyfLZ_1dI?usp=sharing). And then locate the downloaded file as ./checkpoints/ReIN100/model_best.pth.tar

**Related citation(s):**

If you use any materials, please cite the following relevant work(s).

```
@article{FullAttNetNguyen25,
  author       = {Thanh Tuan Nguyen and Hoang Anh Pham and Thinh Vinh Le and Thanh Phuong Nguyen},
  title        = {Full-plane Attentive Patterns Adopted by an Adaptative Lightweight Backbone for Image Recognition},
  journal      = {Transactions on Multimedia},
  note         = {Submitted 2025}
}
```
