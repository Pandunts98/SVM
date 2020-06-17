# ML 101 Homework 4: Support Vector Machine (SVM)

In this homework, you are going to implement Support Vector Machine (SVM) model.


## Kernels

In `kernel.py` file you can find 3 functions of different type of kernels.
First of all you need to implement them. You can find definitions and interesting properties of different kernels in `Bishop`'s book.

## SVM

In `svm.py` there is a class `SVM` having 2 functions you need to implement

 - `gram_matrix` computes `P` matrix (aka `Q` matrix or `H`) 
 - `fit` fits `SVM` model to given data and labels
 - `predict` predicts labels of given data

## Visualize results
 
In order to make sure your model is working correctly, you can use `plot.py`.
Use `python plot.py --help` to watch arguments.

 - `--data-type`: `blobs` or `circles`.`blobs` is simple batch of points, whereas `circles` makes 2 circles one inside another. Use the letter for `guassian` kernel.
 - `--kernel`: 3 types of kernels which you need to implement.
 - `--config`: Config file with json format which includes parameters for kernel and data. For data parameters see `generate_blobs` and `generate_circles` functions inside `plot.py`.
 - `--c`: Regularization parameter.
 
Here is an example
```
python plot.py --data-type blobs --kernel polynomial --config params.json
``` 
Where `params.json` is 
```json
{
  "data": {
    "std": 1,
    "centers": [[1,2], [4,5]]
  },
  "kernel": {
    "p": 5
  }
}
```
![Alt text](./result.png?raw=true "Title")

For any questions or problems about code you can ask on `Google Classrom`.

Good luck :)