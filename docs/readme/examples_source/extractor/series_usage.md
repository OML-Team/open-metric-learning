**Main idea**: The series of images overestimates the metric values (detail explanation below).\
To overcome this you can use our "sequence" functionality.
To use that you need add `sequence` column to [oml_dataset](https://open-metric-learning.readthedocs.io/en/latest/oml/data.html) with the following format:

<img src="https://i.ibb.co/fCqyc6r/Images-Side-By-Side-Static-Manim-CE-v0-18-0.png">

After that calculation of metric will be changed to the following:

<img src="https://i.ibb.co/nwQcqMC/Images-Side-By-Side-Manim-CE-v0-18-0.png">

| metric      | example position | value |
|-------------|------------------|-------|
| CMC@1       | top              | 1.0   |
| CMC@1       | bottom           | 0.0   |
| PRECISION@2 | top              | 0.5   |
| PRECISION@2 | bottom           | 0.5   |

### Why is this important to include series info while counting metrics?
When images in a series come from the same video or a static camera position, they often result in overestimated metric values due to their high similarity. This can make it almost impossible to accurately assess the performance of your model without considering the series information.
For instance, if your validation dataset consists of 1000 classes, with each class containing only 2 images from a single highly correlated series, your model could achieve 100% accuracy on the CMC@1 metric, even with a random model. This underscores the importance of accounting for series-related biases when evaluating model performance.

