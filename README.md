# pyMIToolbox

The pyMIToolbox is inspired by the [MIToolbox](https://github.com/Craigacp/MIToolbox) and [FEAST](https://github.com/Craigacp/FEAST).

Similar to the MIToolbox it provides some functions to calculate the Entropy (H), conditional Entropy, Mutual Information (I), and conditional Mutual Information.
These can be used to impelemt feature selection mechanisms like JMI or HJMI.
For examples, please have a look at the `test` folder.

## Examples
### HJMI feature selection
Historical JMI (HJMI) feature selection mechanism is an extension of the JMI feature selection mechanism. Both deliver the same features if the amount of features is given.
However, the HJMI allows specifying stopping criteria based on the improvement of the overall information of the selected features.
Details can be found in the following paper:

> Gocht, A.; Lehmann, C. & Schöne, R.  
> A New Approach for Automated Feature Selection  
> 2018 IEEE International Conference on Big Data (Big Data), 2018 , 4915-4920  
> [DOI: 10.1109/BigData.2018.8622548](http://doi.org/10.1109/BigData.2018.8622548) or [OA](http://nbn-resolving.de/urn:nbn:de:bsz:14-qucosa2-337156)

For an implementation of the HJMI, please have a loot to  `test/hjmi.py`. Please note, that the algorithm is slightly modified, to avoid a division by zero.

### JMI feature selection

The Joint Mutual Information (JMI) feature selection mechanism is based on the work of Brow et al. and Yang et al.:

> Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection  
> G. Brown, A. Pocock, M.-J. Zhao, M. Lujan  
> Journal of Machine Learning Research, 13:27-66 (2012)  
> [ACM ID: 2188387](http://dl.acm.org/citation.cfm?id=2188385.2188387) or [OA](http://jmlr.csail.mit.edu/papers/v13/brown12a.html)

> Yang, H. H. & Moody, J.  
> Data visualization and feature selection: New algorithms for nongaussian data  
> Advances in Neural Information Processing Systems, 2000 , 687-693

For an implementation of the JMI, please have a loot to  `test/jmi.py`.

## Comparing to MATLAB(R)

Please be aware that Matlab has a different discretisation scheme, then python.

Discretising an Array in python would look like the following:

```
[tmp, features] = X.shape
D = np.zeros([tmp, features])

for i in range(features):
    N, E = np.histogram(X[:,i], bins=10)
    D[:,i] = np.digitize(X[:,i], bins, right=False)
```

While in Matlab the same code would look like:

```
[tmp, features] = size(X);

D=zeros([tmp, features]);
for i = 1:features
    [N,E] = histcounts(X(:,i),10,'BinLimits',[min(X(:,i)),max(X(:,i))]);
    D(:,i) = discretize(X(:,i),E,'IncludedEdge', 'left');
end
```

Moreover, Matlab and Python use different counting for array indexes. While python starts C-like at 0, Matlab starts at 1.
