# Visual Sinkformers - COMP8800 Final Project

This code repository contains the code for the Advanced Computing Project (COMP880) at ANU.

![Visual Transformers Illustration](images/ViT.png)
(*Source: Original ViT paper*)

## Course code
COMP8800: Advanced Computing Research Project

## Research Question
What impact does integrating Sinkformers into Visual Transformer architectures have on image classification accuracy, and how does this enhancement address the limitations of existing methods?

## Relationship with Previous works
This project builds upon recent advancements in Visual Transformers [1] for image classification. While Visual Transformers have shown promising results, they may still face challenges in capturing long-range dependencies efficiently. Sinkformers, a novel attention mechanism introduced in [2], have demonstrated effectiveness in addressing these challenges in various natural language processing and computer vision tasks. By incorporating Sinkformers into Visual Transformers for image classification, this research aims to enhance the model's ability to capture global context and improve classification accuracy.

## Timeline and Milestones

**Note**: These estimated milestones can change depending on the complexity of the task at hand

### Semester 1, 2024
**Week 1 to 4**: Understanding and implementing Sinkformers architecture.
#### Deliverables
- [ ] A Jupyter Notebook with ViT and Sinkformers implementation. 
- [ ] Achieve 79-80% accuracy on the Cats and Dogs image classification dataset, as
demonstrated in the original Sinkformers paper.

**Week 5 to 12**: Incorporating Sinkformers into pre-trained DINOv2 models.
- [ ] A code repository containing the modified (with Sinkformers) DINOv2 models.
- [ ] Fine-tuned DINOv2 models on the ImageNet-1k dataset, with calculated top-1
accuracy results and relevant result interpretation.

## Resources
Resources used to implement the project
- [Multi-head attention from scratch](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)
- [Model training on AWS](https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart)
- [Push from AWS to Github](https://www.reddit.com/r/aws/comments/13zb1gm/how_to_push_to_remote_github_repository_from_aws/)
- [Sinkhorn-Knopp algorithm](https://fulkast.medium.com/the-sinkhorn-knopp-algorithm-without-proof-697c9af7df7)
- [Sinkhorn AutoDiff](https://arxiv.org/pdf/1706.00292.pdf)
- [Sinkhorn AutoDiff Algorithm](https://arxiv.org/pdf/1607.05816.pdf)
