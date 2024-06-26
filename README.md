# Mitigating annotation shift in cancer classification using single image generative models

## Citation

```
Marta Buetas Arcas, Richard Osuala, Karim Lekadir, Oliver Díaz, "Mitigating annotation shift in cancer classification using single-image generative models," Proc. SPIE 13174, 17th International Workshop on Breast Imaging (IWBI 2024), 1317421 (29 May 2024); https://doi.org/10.1117/12.3025548
```
- Link to SPIE publication: https://doi.org/10.1117/12.3025548
- Link to arXiv: https://arxiv.org/abs/2405.19754
  
This repository contains all the code and sources used for the development of the experiments conducted in the project, accepted in the SPIE International Workshop on Breast Imaging (IWBI 2024) conference.

## Abstract

Artificial Intelligence (AI) has emerged as a beneficial tool to assist radiologists in breast cancer detection and diagnosis. Both the quantity and quality of available data have a direct impact on the success of these applications. One major challenge is the scarcity of labeled data, largely attributable to the extensive time, effort, and costs associated with acquiring expert annotations. This often limits the training and evaluation of AI models, causing a lack of generalisation and robustness. This issue is further exacerbated by the varying quality of available expert annotations that commonly display high intra- and inter-observer variability. This can lead to annotation shift, where a model’s performance decreases at test time if test annotations differ from their training counterparts, for example, in size, accuracy, delineation, lesion boundary and margin definition, annotation protocol or sourcing modality.

**Our approach:**
To increase classification model robustness against annotation shift, one approach is to generate additional training images that correspond to the desired annotation characteristics of the target domain (in-domain). To this end, we propose the selection of a single well-annotated in-domain training image to train a generative AI model, which, in turn, learns to synthesize an arbitrary number of variations. These variations are readily usable as additional classification model training images.

## Materials

<div align="center">
    <img src="https://github.com/MartaBuetas/IWBIconference_EnhancingBreastCancerDiagnosis/assets/101974217/725899a7-660e-4dfc-bdb1-89bacb19f853", alt="Conference Image" width="500">
</div>

*Figure 1.* Our study's primary goal. The classification task in this study is to obtain a pre-biopsy cancer risk prediction of breast lesions. Our multiclass deep-learning classification model distinguishes between (i) healthy tissue, (ii) benign, and (iii) malignant lesions, thereby extending on previous approaches, which focused on general healthy vs non-healthy classification. 

### Data Preparation and Preprocessing

- The classifier works at patch level and these patches were extracted from the mammograms, including both lesion and healthy areas, in both scanned and digital formats. The technique followed for generating the patch dataset is explained in detail in the Python notebook `generate_patch_dataset.ipynb`. To generate the dataset, the [BCDR dataset](https://bcdr.eu/) needs to be downloaded. Three metadata .csv files are also generated, one for lesions, another for healthy digital patches, and a third one for scanned film patches. These files contain the corresponding data required for the study objectives, each with a unique ID for each patch. To convert the previously generated .csv files into a unified .jsonl metadata file, the Python script `csv_to_jsonl_metadata.py` is used.

- To **simulate annotation shift**, we extract patches from the lesions from more and less tightly fitting bounding boxes surrounding the lesion, i.e., with different levels of zoom. In practice, an accurate lesion delineation allows to extract a tight lesion bounding box. On the other hand, rectangle lesion annotations (e.g. performed either by human experts or by object detection models) contain varying amounts of healthy tissue surrounding the lesions. Therefore, our bounding boxes -- extracted based on different zoom levels -- simulate varying annotation protocols (annotation shift) and thus allow to measure their influence on classification performance. Thus, for each lesion, three patches are defined and extracted with different levels of zoom, capturing varying percentages of adjacent healthy tissue. Group 1 (G1) patches correspond to the most accurate bounding box defined around the original annotated lesion delineation mask. Group 2 (G2) and 3 (G3) capture patches with double (200\%) and triple (300\%) the height and width of the original bounding box, respectively.
<div align="center">
    <img src="https://github.com/MartaBuetas/IWBIconference_EnhancingBreastCancerDiagnosis/assets/101974217/f697bac0-9aba-4663-a7ae-c8db4fb012b7" alt="Conference Image" width="500">
</div>

*Figure 2.* Digital mammogram with a biopsy-proven malignant lesion and its corresponding lesion annotation mask. The extracted patches are depicted with increasing extend of non-lesion tissue visible on the patch.


### Patch Classifier Configuration

- The first contribution in this research is the design and implementation of a high-accuracy malignancy classification model trained to distinguish cancerous from benign breast lesions. As classification model, a ResNet50 was used, which we initialise with weights pretrained on the ImageNet11 dataset, available at [PyTorch: models and pre-trained weights](https://pytorch.org/vision/stable/models.html). To optimise the training process, only the parameters of the last layer were kept trainable. For the multiclass task, there were finally 6147 trainable parameters.

- Each experiment in this study ran for 100 epochs and the model from the epoch with the best validation loss was selected. Models are evaluated based on a train-validation-test split across three folds, ensuring that each patient was present in only one of the sets. The data was partitioned into 10\% for testing (344 samples), 63\% for training (2167 samples), and 27\% for validation (929 samples). The experiments were conducted with consistent hyperparameters to ensure fair comparisons between methods. These included a fixed batch size of 128, utilizing the adaptive moment optimiser (Adam) with default beta parameters ($\beta_1$=0.9 and $\beta_2$=0.999), and employing a learning rate scheduler that gradually decayed the learning rate which started at 1e-2. The scheduler had a step of 5 epochs and a gamma value of 0.1. For both the binary classification problem and the multiclass task, a binary cross-entropy loss function was employed. All experiments were run on a NVIDIA RTX 2080 Super 8GB GPU using the PyTorch library. Training the classifier for 100 epochs took approximately 3 hours in this setup. across 3 folds.

- We initially assessed a binary classification task distinguishing between healthy and lesion-containing patches, yielding a test accuracy of 0.924 ± 0.009 and a test ROC-AUC (Area under the Receiver Operating Characteristic Curve) of 0.971 ± 0.009, indicating how effectively the model classifies this binary class. The `binary_pipeline.py` script contains the code required for training and testing the classifier. Subsequent experiments extended this setup to multiclass classification, categorizing patches as healthy, benign, or malignant. The `train_classifier_[...].py` scripts are dedicated to the multiclass task. Changes from the binary script include employing Categorical Cross Entropy as the loss function and adjusting the dimensions of the last layer in the ResNet50 model. Moreover, we incorporated options for augmenting training data using various methods.

### SinGAN Model Training

We investigate data augmentation using SinGAN models to enhance classifier performance under annotation shift for the 'malignant' class. This class showed this consistently the lowest classification performance in previous experiments and it was also heavily impacted by annotation shift. Each of the SinGAN models used for generating new data was trained exclusively on malignant lesion patches from a specific level of zoom (e.g. G1). We note that G1 patches represent the most accurate lesion annotations and, thus, are the most challenging to obtain. For instance, G2 or G3 lesion patches can be retrieved from a G1 annotation, but not vice versa, as the lesion of a G2 or G3 annotation can be (partly) outside the boundaries of a cropped (zoomed-in) G1 patch.

- The folder `SinGAN` includes all the required files for training a SinGAN model from a specific training image, generating synthetic samples, and validating them through the SiFID metric presented by [Shaham, Dekel, and Michaeli](https://ui.adsabs.harvard.edu/abs/2019arXiv190501164R/abstract). It is a clone of [the official pytorch implementation of the paper: "SinGAN: Learning a Generative Model from a Single Natural Image"](https://github.com/tamarott/SinGAN), with adaptations for hyperparameter setup and some additional files for the convenience of the development fo this project. For each scale, the final model after 30 training epochs was selected. The default values were used for the rest of the hyperparameters. Firstly, the script `input_preprocess.py` was added to preprocess the patches, as it is needed as input for the SinGAN model. Secondly, the script `generating_singan.py` was added to train the desired SinGAN models using the preprocessed patches and generate new images from the trained model. The original `config.py` file was modified to have a pyramid scale factor of 0.8. Therefore, the resolution of the image is reduced by 20% when passing to the next scale. To quantitatively asses synthetic data quality, we use the Single Image FID (SiFID) metric  measuring the similarity between the real and synthetic images. With the notebook `SIFID_evaluation.ipynb` the SiFID metric for the generated samples can be computed.

### Integration of SinGAN Generated Images

- Multiple SinGAN models are trained individually, each on a distinct single malignant lesion image. Synthetic datasets are then assembled, ensuring each SinGAN contributes an equal number of samples to balance the training dataset of the classification model, maintaining an equal number of benign and malignant lesions. These synthetic datasets, varying in the number of SinGANs used, serve as augmentation data during classifier training. This systematic approach enables the study of the impact of different numbers of SinGAN models on synthetic dataset generation. The number of SinGAN models used varied between 1, 2, and 4. Consequently, model performance was compared when trained with data augmented using different numbers of SinGAN models. For clarity, separate files are created for each scenario, such as `train_classifier_2SinGANs_multiclass.py`, enabling augmentation with synthetic samples from 2 SinGAN models, with similar files for 1 and 4 SinGAN models. The most favorable outcome for the malignant class in terms of ROC-AUC was achieved when generating new data from 4 different SinGAN models. The main conclusions of this set of experiments were:
- Using 1 to 4 lesion images with in-domain annotations, SinGAN data augmentation improves the baseline for the malignant class.
- The choice of images for training SinGAN likely has a significant impact on the classifier's performance.

### Comparison with Traditional Data Augmentation technique

- Data augmentation using SinGAN-generated samples is compared and combined with traditional methods, specifically single-image (G1) oversampling. Experiments with SinGAN-augmented data involve generating a synthetic dataset from 4 SinGAN models trained on different lesion images from group G1. For consistency, the same four training images utilized for training each SinGAN model are also employed for the oversampling method. Additionally, experiments can be run with random resized cropping applied to test samples to measure classification robustness in cases where the lesion is not at the center of the region-of-interest image.

### Ensemble Architecture and Performance Evaluation

- In all multiclass classifier experiments, an ensemble architecture is utilized. For both augmentation techniques—SinGAN-based and oversampling—and for the baseline (no data augmentation), three models are trained with a train-validation split across three folds, ensuring each patient appears in only one set. Evaluation employs a fixed test set comprising images from patients exclusively allocated to this set. Ensemble prediction is determined as the mean of predictions generated by each of the three models within the ensemble. In the case of the ensemble combining SinGAN augmentation and oversampling, a total of six models are configured for the ensemble. Notably, the test set remains unchanged across all experiments. Once classifiers are trained, the script `test_trained_models.py` can be used to test the classifier using different methods specified by keyword input flags: oversampling technique, SinGAN-based data augmentation, an ensemble of both techniques, or the baseline with no augmentation.

<div align="center">
    <img src="https://github.com/MartaBuetas/IWBIconference_EnhancingBreastCancerDiagnosis/assets/101974217/383c906d-9589-459b-85be-5dbbc934414b", alt="Conference Image" width="600">
</div>

*Figure 3.* General pipeline of our experiments. Patches from healthy and lesion samples are extracted from the BCDR dataset at three zoom levels (G1, G2, G3) per lesion. The dataset is split into three folds, ensuring images from the same patient are in a single set. G1 patches train different SinGAN 
models individually. Generated samples create a synthetic dataset to balance the training data.


