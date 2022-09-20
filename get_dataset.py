
from ogb.lsc import MAG240MDataset
dataset = MAG240MDataset(root="dataset_path")

print(dataset.num_papers) # number of paper nodes
print(dataset.num_authors) # number of author nodes
print(dataset.num_institutions) # number of institution nodes
print(dataset.num_paper_features) # dimensionality of paper features
print(dataset.num_classes) # number of subject area classes

print(dataset.paper_label) # numpy array of shape (num_papers, ), storing target labels of papers.