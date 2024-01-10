# Final

### Round 1
- Accuracy : 0.95
- Rank : 1
### Round 2
- Accuracy : 0.97
- Rank : 1
### Final Round
- Accuracy : 0.9670
- Rank : 1

## Configuration instructions
1. KNN
```python
knn = KNeighborsClassifier(n_neighbors=1)
```
2. SVM
```python
svc = Pipeline([
    ('pca', PCA(n_components=800)),
    ('MM', MinMaxScaler()),
    ('svm', SVC(C=100, kernel='rbf', degree=2, gamma=50, probability=True, random_state=22))
])
```

3. Voting
```python
pipeline = VotingClassifier(estimators=classifiers,
                             voting='soft',
                             weights=[100, 3])
```


## Copyright and licensing information
- MIT License

## Contact information
- Name: Seongryong Jung
- E-mail: jungsr1116@cau.ac.kr
