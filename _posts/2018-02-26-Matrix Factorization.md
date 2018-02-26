
# **What are we doing?**

We are building a basic version of a low-rank matrix factorization recommendation system and we will use it on a dataset obtained from https://grouplens.org/datasets/movielens/. It has 20 million ratings and 465,000 tag applications applied to 27,000 movies by 138,000 users. 

The other technique to build a recommendation system is an item-based collaborative filtering approach. Collaborative filtering methods that compute distance relationships between items or users are generally thought of as "neighborhood" methods, since they are centered on the idea of "nearness". 

These methods are not well-suited for larger datasets. There is another conceptual issue with them as well, i.e., the ratings matrices may be overfit and noisy representations of user tastes and preferences.When we use distance based "neighborhood" approaches on raw data, we match to sparse low-level details that we assume represent the user's preference vector instead of the vector itself. It's a subtle difference, but it's important.

If I've listened to ten Breaking Benjamin songs and you've listened to ten different Breaking Benjamin songs, the raw user action matrix wouldn't have any overlap. We'd have nothing in common, even though it seems pretty likely we share at least some underlying preferencs. We need a method that can derive the tastes and preference vectors from the raw data.

Low-Rank Matrix Factorization is one of those methods.

# **Basics of Matrix Factorization for Recommendation systems**

#### All of the theoretical explanation has been taken from: http://nicolas-hug.com/blog/matrix_facto_1

#### I could have written it myself, but I loved the explanation in the link and I would love it if people read it completely to understand how Matrix Factorization for recommendation systems actally works

The problem we need to assess is that of rating prediction. The data we would have on our hands is a **rating history**.

It would look something like this:

![no-alignment]({{ site.url }}{{ site.baseurl }}/assets/images/matrixfactorization/Rmatrix.JPG)

Our **R matrix** is a 99% sparse matrix withthe columns as the items or movies in our case and the rows as individual users.

We will factorize the matrix R. The matrix factorization is linked to SVD(Singular Value Decomposition). It's a beautiful result of Linear Algebra. When people say Math sucks, show them what SVD can do.

But, before we move onto SVD, we should review PCA(Principle Components Analysis). It's only slightly less awesome than SVD, but it's still pretty cool.

# **A little bit of PCA**

We’ll play around with the Olivetti dataset. It’s a set of greyscale images of faces from 40 people, making up a total of 400 images.


```python
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
%matplotlib inline
```


```python
faces = fetch_olivetti_faces()
print(faces.DESCR)
```

    Modified Olivetti faces dataset.
    
    The original database was available from
    
        http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
    
    The version retrieved here comes in MATLAB format from the personal
    web page of Sam Roweis:
    
        http://www.cs.nyu.edu/~roweis/
    
    There are ten different images of each of 40 distinct subjects. For some
    subjects, the images were taken at different times, varying the lighting,
    facial expressions (open / closed eyes, smiling / not smiling) and facial
    details (glasses / no glasses). All the images were taken against a dark
    homogeneous background with the subjects in an upright, frontal position (with
    tolerance for some side movement).
    
    The original dataset consisted of 92 x 112, while the Roweis version
    consists of 64x64 images.
    
    

**Here are the first 10 people:**


```python
# Here are the first ten guys of the dataset
fig = plt.figure(figsize=(10, 10))
for i in range(10):
    ax = plt.subplot2grid((1, 10), (0, i))
    
    ax.imshow(faces.data[i * 10].reshape(64, 64), cmap=plt.cm.gray)
    ax.axis('off')
```


![no-alignment]({{ site.url }}{{ site.baseurl }}/assets/images/matrixfactorization/output_11_0.png)


**Each image size is 64 x 64 pixels. We will flatten each of these images (we thus get 400 vectors, each with 64 x 64 = 4096 elements). We can represent our dataset in a 400 x 4096 matrix:**

insert flattened

**PCA, which stands for Principal Component Analysis, is an algorithm that will reveal 400 of these guys:**


```python
# Let's compute the PCA
pca = PCA()
pca.fit(faces.data)
```




    PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)




```python
# Now, the creepy guys are in the components_ attribute.
# Here are the first ten ones:

fig = plt.figure(figsize=(10, 10))
for i in range(10):
    ax = plt.subplot2grid((1, 10), (0, i))
    
    ax.imshow(pca.components_[i].reshape(64, 64), cmap=plt.cm.gray)
    ax.axis('off')
```

![no-alignment]({{ site.url }}{{ site.baseurl }}/assets/images/matrixfactorization/output_15_0.png)


**This is pretty creepy, right?**

We call these guys the principal components (hence the name of the technique), and when they represent faces such as here we call them the eigenfaces. Some really cool stuff can be done with eigenfaces such as face recognition, or optimizing your tinder matches! The reason why they’re called eigenfaces is because they are in fact the eigenvectors of the covariance matrix of X

We obtain here 400 principal components because the original matrix X has 400 rows (or more precisely, because the rank of X is 400). As you may have guessed, each of the principal component is in fact a vector that has the same dimension as the original faces, i.e. it has 64 x 64 = 4096 pixels.


```python
# Reconstruction process

from skimage.io import imsave

face = faces.data[0]  # we will reconstruct the first face

# During the reconstruction process we are actually computing, at the kth frame,
# a rank k approximation of the face. To get a rank k approximation of a face,
# we need to first transform it into the 'latent space', and then
# transform it back to the original space

# Step 1: transform the face into the latent space.
# It's now a vector with 400 components. The kth component gives the importance
# of the kth  creepy guy
trans = pca.transform(face.reshape(1, -1))  # Reshape for scikit learn

# Step 2: reconstruction. To build the kth frame, we use all the creepy guys
# up until the kth one.
# Warning: this will save 400 png images.
for k in range(400):
    rank_k_approx = trans[:, :k].dot(pca.components_[:k]) + pca.mean_
    imsave('{:>03}'.format(str(k)) + '.jpg', rank_k_approx.reshape(64, 64))
```

    E:\Software\Anaconda2\envs\py36\lib\site-packages\skimage\util\dtype.py:122: UserWarning: Possible precision loss when converting from float64 to uint8
      .format(dtypeobj_in, dtypeobj_out))
    

As far as we’re concerned, we will call these guys the **creepy guys**. 

Now, one amazing thing about them is that they can build back all of the original faces. Take a look at this (these are animated gifs, about 10s long):

Insert gifs

Each of the 400 original faces (i.e. each of the 400 original rows of the matrix) can be expressed as a (linear) combination of the creepy guys. That is, we can express the first original face (i.e. its pixel values) as a little bit of the first creepy guy, plus a little bit of the second creepy guy, plus a little bit of third, etc. until the last creepy guy. The same goes for all of the other original faces: they can all be expressed as a little bit of each creepy guy.

**Face 1 = α1⋅Creepy guy #1 + α2⋅Creepy guy #2 . . . + α400⋅Creepy guy #400**

The gifs you saw above are the very translation of these math equations: the first frame of a gif is the contribution of the first creepy guy, the second frame is the contribution of the first two creepy guys, etc. until the last creepy guy.

### Latent Factors

We’ve actually been kind of harsh towards the creepy guys. They’re not creepy, they’re typical. The goal of PCA is to reveal typical vectors: each of the creepy/typical guy represents one specific aspect underlying the data. In an ideal world, the first typical guy would represent (e.g.) a typical elder person, the second typical guy would represent a typical glasses wearer, and some other typical guys would represent concepts such as smiley, sad looking, big nose, stuff like that. And with these concepts, we could define a face as more or less elder, more or less glassy, more or less smiling, etc. In practice, the concepts that PCA reveals are really not that clear: there is no clear semantic that we could associate with any of the creepy/typical guys that we obtained here. But the important fact remains: each of the typical guys captures a specific aspect of the data. We call these aspects the latent factors (latent, because they were there all the time, we just needed PCA to reveal them). Using barbaric terms, we say that each principal component (the creepy/typical guys) captures a specific latent factor.

Now, this is all good and fun, but we’re interested in matrix factorization for recommendation purposes, right? So where is our matrix factorization, and what does it have to do with recommendation? PCA is actually a plug-and-play method: it works for any matrix. If your matrix contains images, it will reveal some typical images that can build back all of your initial images, such as here. If your matrix contains potatoes, PCA will reveal some typical potatoes that can build back all of your original potatoes. If your matrix contains ratings, well… Here we come.

# PCA on a (dense) rating matrix

Until stated otherwise, we will consider for now that our rating matrix R is completely dense, i.e. there are no missing entries. All the ratings are known. This is of course not the case in real recommendation problems, but bare with me.

### PCA on the users

Here is our rating matrix, where rows are users and columns are movies:

Insert userPCA

Instead of having faces in the rows represented by pixel values, we now have users represented by their ratings. Just like PCA gave us some typical guys before, it will now give us some typical users, or rather some typical raters.

we would obtain a typical action movie fan, a typical romance movie fan, a typical comedy fan, etc. In practice, the semantic behind the typical users is not clearly defined, but for the sake of simplicity we will assume that they are (it doesn’t change anything, this is just for intuition/explanation purposes).

Each of our initial users (Alice, Bob…) can be expressed as a combination of the typical users. For instance, Alice could be defined as a little bit of an action fan, a little bit of a comedy fan, a lot of a romance fan, etc. As for Bob, he could be more keen on action movies:

**Alice = 10% Action fan + 10% Comedy fan + 50% Romance fan + ...**

**Bob = 50% Action fan + 30% Comedy fan + 10% Romance fan + ...**

And the same goes for all of the users, you get the idea. (In practice the coefficients are not necessarily percentages, but it’s convenient for us to think of it this way).

### PCA on the movies

What would happen if we transposed our rating matrix? Instead of having users in the rows, we would now have movies, defined as their ratings:

insert moviesPCA

In this case, PCA will not reveal typical faces nor typical users, but of course typical movies. And here again, we will associate a semantic meaning behind each of the typical movies, and these typical movies can build back all of our original movies:

And the same goes for all the other movies.

So what can SVD do for us? SVD is PCA on R and R(Transpose), in one shot.

SVD will give you the two matrices U and M, at the same time. You get the typical users and the typical movies in one shot. SVD gives you U and M by factorizing R into three matrices. Here is the matrix factorization:

**R=MΣU(Transpose)**

To be very clear: SVD is an algorithm that takes the matrix R as an input, and it gives you M, Σ and U, such that:

R is equal to the product **MΣU(Transpose).**

The columns of M can build back all of the columns of R (we already know this).

The columns of U can build back all of the rows of R (we already know this).

The columns of M are orthogonal, as well as the columns of U. I haven’t mentioned this before, so here it is: the principal components are always orthogonal. This is actually an extremely important feature of PCA (and SVD), but for our recommendation we actually don’t care (we’ll come to that).

Σ is a diagonal matrix (we’ll also come to that).

We can basically sum up all of the above points by this statements: the columns of M are an orthogonal basis that spans the column space of R, and the columns of U are an orthonormal basis that spans the row space of R. If this kind of phrases works for you, great. Personally, I prefer to talk about creepy guys and typical potatoes

### The model behind SVD 

When we compute and use the SVD of the rating matrix R, we are actually modeling the ratings in a very specific, and meaningful way. We will describe this modeling here.

For the sake of simplicity, we will forget about the matrix Σ: it is a diagonal matrix, so it simply acts as a scaler on M or U(Transpose). Hence, we will pretend that we have merged into one of the two matrices. Our matrix factorization simply becomes:

R=MU(Transpose)

Now, with this factorization, let’s consider the rating of user u for item i, that we will denote rui:

Insert productmatrices

Because of the way a matrix product is defined, the value of rui is the result of a dot product between two vectors: a vector pu which is a row of M and which is specific to the user u, and a vector qi which is a column of UT and which is specific to the item i:

rui=pu⋅qi,

where '⋅' stands for the usual dot product. Now, remember how we can describe our users and our items?

**Alice = 10% Action fan + 10% Comedy fan + 50% Romance fan + ...**

**Bob = 50% Action fan + 30% Comedy fan + 10% Romance fan + ...**

**Titanic = 20% Action + 0% Comedy + 70% Romance + ...**

**Toy Story = 30% Action + 60% Comedy + 0% Romance + ...**

Well, the values of the vectors pu and qi exactly correspond to the coefficients that we have assigned to each latent factor:

**pAlice=(10%,  10%,  50%, ...)**

**pBob=(50%,  30%,  10%, ...)**

**qTitanic=(20%,  0%,  70%, ...)**

**qToy Story=(30%,  60%,  0%, ...)**

The vector pu represents the affinity of user u for each of the latent factors. Similarly, the vector qi represents the affinity of the item i for the latent factors. Alice is represented as (10%,10%,50%,...), meaning that she’s only slightly sensitive to action and comedy movies, but she seems to like romance. As for Bob, he seems to prefer action movies above anything else. We can also see that Titanic is mostly a romance movie and that it’s not funny at all.

So, when we are using the SVD of R, we are modeling the rating of user u for item i as follows:

insert equation

n other words, if u has a taste for factors that are endorsed by i, then the rating rui will be high. Conversely, if i is not the kind of items that u likes (i.e. the coefficient don’t match well), the rating rui will be low. In our case, the rating of Alice for Titanic will be high, while that of Bob will be much lower because he’s not so keen on romance movies. His rating for Toy Story will, however, be higher than that of Alice.

We now have enough knowledge to apply SVD to a recommendation task.

# **Setting up the ratings data**


```python
import pandas as pd
import numpy as np
```


```python
movies_df=pd.read_csv("E:/Git/Project markdowns/Matrix Factorization/ml-20m/movies.csv")
```


```python
movies_df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Heat (1995)</td>
      <td>Action|Crime|Thriller</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Sabrina (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Tom and Huck (1995)</td>
      <td>Adventure|Children</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Sudden Death (1995)</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>GoldenEye (1995)</td>
      <td>Action|Adventure|Thriller</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings_df=pd.read_csv("E:/Git/Project markdowns/Matrix Factorization/ml-20m/ratings.csv")
```


```python
ratings_df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3.5</td>
      <td>1112486027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>29</td>
      <td>3.5</td>
      <td>1112484676</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>32</td>
      <td>3.5</td>
      <td>1112484819</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>3.5</td>
      <td>1112484727</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>3.5</td>
      <td>1112484580</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>112</td>
      <td>3.5</td>
      <td>1094785740</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>151</td>
      <td>4.0</td>
      <td>1094785734</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>223</td>
      <td>4.0</td>
      <td>1112485573</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>253</td>
      <td>4.0</td>
      <td>1112484940</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>260</td>
      <td>4.0</td>
      <td>1112484826</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Defining a list of numbers from 0 to 7999
mylist=range(0, 8001)
```


```python
#Getting the data for the first 8000 users
train_df=ratings_df[ratings_df["userId"].isin(mylist)]
```


```python
train_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3.5</td>
      <td>1112486027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>29</td>
      <td>3.5</td>
      <td>1112484676</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>32</td>
      <td>3.5</td>
      <td>1112484819</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>3.5</td>
      <td>1112484727</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>3.5</td>
      <td>1112484580</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>112</td>
      <td>3.5</td>
      <td>1094785740</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>151</td>
      <td>4.0</td>
      <td>1094785734</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>223</td>
      <td>4.0</td>
      <td>1112485573</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>253</td>
      <td>4.0</td>
      <td>1112484940</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>260</td>
      <td>4.0</td>
      <td>1112484826</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>293</td>
      <td>4.0</td>
      <td>1112484703</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>296</td>
      <td>4.0</td>
      <td>1112484767</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>318</td>
      <td>4.0</td>
      <td>1112484798</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>337</td>
      <td>3.5</td>
      <td>1094785709</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>367</td>
      <td>3.5</td>
      <td>1112485980</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>541</td>
      <td>4.0</td>
      <td>1112484603</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>589</td>
      <td>3.5</td>
      <td>1112485557</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>593</td>
      <td>3.5</td>
      <td>1112484661</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>653</td>
      <td>3.0</td>
      <td>1094785691</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>919</td>
      <td>3.5</td>
      <td>1094785621</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>924</td>
      <td>3.5</td>
      <td>1094785598</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>1009</td>
      <td>3.5</td>
      <td>1112486013</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>1036</td>
      <td>4.0</td>
      <td>1112485480</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>1079</td>
      <td>4.0</td>
      <td>1094785665</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>1080</td>
      <td>3.5</td>
      <td>1112485375</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>1089</td>
      <td>3.5</td>
      <td>1112484669</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>1090</td>
      <td>4.0</td>
      <td>1112485453</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>1097</td>
      <td>4.0</td>
      <td>1112485701</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>1136</td>
      <td>3.5</td>
      <td>1112484609</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>1193</td>
      <td>3.5</td>
      <td>1112484690</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1170896</th>
      <td>8000</td>
      <td>858</td>
      <td>5.0</td>
      <td>1360263827</td>
    </tr>
    <tr>
      <th>1170897</th>
      <td>8000</td>
      <td>1073</td>
      <td>5.0</td>
      <td>1360263822</td>
    </tr>
    <tr>
      <th>1170898</th>
      <td>8000</td>
      <td>1092</td>
      <td>4.5</td>
      <td>1360262914</td>
    </tr>
    <tr>
      <th>1170899</th>
      <td>8000</td>
      <td>1093</td>
      <td>3.5</td>
      <td>1360263017</td>
    </tr>
    <tr>
      <th>1170900</th>
      <td>8000</td>
      <td>1136</td>
      <td>4.0</td>
      <td>1360263818</td>
    </tr>
    <tr>
      <th>1170901</th>
      <td>8000</td>
      <td>1193</td>
      <td>4.5</td>
      <td>1360263372</td>
    </tr>
    <tr>
      <th>1170902</th>
      <td>8000</td>
      <td>1213</td>
      <td>4.0</td>
      <td>1360263806</td>
    </tr>
    <tr>
      <th>1170903</th>
      <td>8000</td>
      <td>1293</td>
      <td>3.0</td>
      <td>1360262929</td>
    </tr>
    <tr>
      <th>1170904</th>
      <td>8000</td>
      <td>1333</td>
      <td>4.0</td>
      <td>1360262893</td>
    </tr>
    <tr>
      <th>1170905</th>
      <td>8000</td>
      <td>1405</td>
      <td>2.5</td>
      <td>1360262963</td>
    </tr>
    <tr>
      <th>1170906</th>
      <td>8000</td>
      <td>1544</td>
      <td>0.5</td>
      <td>1360263394</td>
    </tr>
    <tr>
      <th>1170907</th>
      <td>8000</td>
      <td>1645</td>
      <td>4.0</td>
      <td>1360262901</td>
    </tr>
    <tr>
      <th>1170908</th>
      <td>8000</td>
      <td>1673</td>
      <td>4.0</td>
      <td>1360263436</td>
    </tr>
    <tr>
      <th>1170909</th>
      <td>8000</td>
      <td>1732</td>
      <td>4.5</td>
      <td>1360263799</td>
    </tr>
    <tr>
      <th>1170910</th>
      <td>8000</td>
      <td>1884</td>
      <td>3.5</td>
      <td>1360263556</td>
    </tr>
    <tr>
      <th>1170911</th>
      <td>8000</td>
      <td>2371</td>
      <td>4.0</td>
      <td>1360263061</td>
    </tr>
    <tr>
      <th>1170912</th>
      <td>8000</td>
      <td>2539</td>
      <td>3.0</td>
      <td>1360262950</td>
    </tr>
    <tr>
      <th>1170913</th>
      <td>8000</td>
      <td>3255</td>
      <td>3.5</td>
      <td>1360262934</td>
    </tr>
    <tr>
      <th>1170914</th>
      <td>8000</td>
      <td>3671</td>
      <td>4.5</td>
      <td>1360263492</td>
    </tr>
    <tr>
      <th>1170915</th>
      <td>8000</td>
      <td>3717</td>
      <td>3.5</td>
      <td>1360262956</td>
    </tr>
    <tr>
      <th>1170916</th>
      <td>8000</td>
      <td>3911</td>
      <td>4.5</td>
      <td>1360262942</td>
    </tr>
    <tr>
      <th>1170917</th>
      <td>8000</td>
      <td>5669</td>
      <td>3.0</td>
      <td>1360263898</td>
    </tr>
    <tr>
      <th>1170918</th>
      <td>8000</td>
      <td>6863</td>
      <td>4.0</td>
      <td>1360263887</td>
    </tr>
    <tr>
      <th>1170919</th>
      <td>8000</td>
      <td>7836</td>
      <td>4.5</td>
      <td>1360263540</td>
    </tr>
    <tr>
      <th>1170920</th>
      <td>8000</td>
      <td>8376</td>
      <td>4.0</td>
      <td>1360263882</td>
    </tr>
    <tr>
      <th>1170921</th>
      <td>8000</td>
      <td>8622</td>
      <td>2.5</td>
      <td>1360263878</td>
    </tr>
    <tr>
      <th>1170922</th>
      <td>8000</td>
      <td>8917</td>
      <td>4.0</td>
      <td>1360263478</td>
    </tr>
    <tr>
      <th>1170923</th>
      <td>8000</td>
      <td>35836</td>
      <td>4.0</td>
      <td>1360263689</td>
    </tr>
    <tr>
      <th>1170924</th>
      <td>8000</td>
      <td>50872</td>
      <td>4.5</td>
      <td>1360263854</td>
    </tr>
    <tr>
      <th>1170925</th>
      <td>8000</td>
      <td>55290</td>
      <td>4.0</td>
      <td>1360263330</td>
    </tr>
  </tbody>
</table>
<p>1170926 rows × 4 columns</p>
</div>




```python
#Time to define the R matrix that we discussed about earlier
R_df = train_df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
```


```python
R_df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>movieId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>129822</th>
      <th>129857</th>
      <th>130052</th>
      <th>130073</th>
      <th>130219</th>
      <th>130462</th>
      <th>130490</th>
      <th>130496</th>
      <th>130642</th>
      <th>130768</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 14365 columns</p>
</div>



**The last thing we need to do is de-mean the data (normalize by each users mean) and convert it from a dataframe to a numpy array.**

**With my ratings matrix properly formatted and normalized, I would be ready to do the singular value decomposition.**


```python
R = R_df.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
R_demeaned
```




    array([[-0.04559694,  3.45440306, -0.04559694, ..., -0.04559694,
            -0.04559694, -0.04559694],
           [-0.01698573, -0.01698573,  3.98301427, ..., -0.01698573,
            -0.01698573, -0.01698573],
           [ 3.94632788, -0.05367212, -0.05367212, ..., -0.05367212,
            -0.05367212, -0.05367212],
           ..., 
           [ 2.91806474, -0.08193526, -0.08193526, ..., -0.08193526,
            -0.08193526, -0.08193526],
           [-0.01141664, -0.01141664, -0.01141664, ..., -0.01141664,
            -0.01141664, -0.01141664],
           [-0.01127741, -0.01127741, -0.01127741, ..., -0.01127741,
            -0.01127741, -0.01127741]])



**Scipy and Numpy both have functions to do the singular value decomposition. We would be using the Scipy function svds because it let's us choose how many latent factors we want to use to approximate the original ratings matrix.**


```python
from scipy.sparse.linalg import svds
M, sigma, Ut = svds(R_demeaned, k = 50)
```

**The function returns exactly those matrices detailed earlier in this post, except that the $\Sigma$ returned is just the values instead of a diagonal matrix. So, we will convert those numbers into a diagonal matrix.**


```python
sigma = np.diag(sigma)
```

# **Time for Predicting**

**We would be adding the user means to the data to get the original means back**


```python
allpredictedratings = np.dot(np.dot(M, sigma), Ut) + user_ratings_mean.reshape(-1, 1)
```

To put this kind of a system into production, we would have to create a training and validation set and optimize the number of latent features ($k$) by minimizing the Root Mean Square Error. Intuitively, the Root Mean Square Error will decrease on the training set as $k$ increases (because I'm approximating the original ratings matrix with a higher rank matrix).

For movies, between 20 and 100 feature "preferences" vectors have been found to be optimal for generalizing to unseen data.

We won't be optimizing the $k$ for this post.

# Giving the movie Recommendations

With the predictions matrix for every user, we can define a function to recommend movies for any user. All we need to do is return the movies with the highest predicted rating that the specified user hasn't already rated.

We will also return the list of movies the user has already rated


```python
predicted_df = pd.DataFrame(allpredictedratings, columns = R_df.columns)
predicted_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>movieId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>129822</th>
      <th>129857</th>
      <th>130052</th>
      <th>130073</th>
      <th>130219</th>
      <th>130462</th>
      <th>130490</th>
      <th>130496</th>
      <th>130642</th>
      <th>130768</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.469187</td>
      <td>0.766043</td>
      <td>0.175666</td>
      <td>0.020263</td>
      <td>-0.144290</td>
      <td>0.128583</td>
      <td>-0.347741</td>
      <td>0.011184</td>
      <td>-0.166920</td>
      <td>0.016176</td>
      <td>...</td>
      <td>-0.001710</td>
      <td>-0.009119</td>
      <td>-0.003445</td>
      <td>0.001706</td>
      <td>-0.011580</td>
      <td>-0.009186</td>
      <td>0.001071</td>
      <td>-0.000878</td>
      <td>-0.005326</td>
      <td>-0.002008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.059213</td>
      <td>0.008119</td>
      <td>0.334548</td>
      <td>0.095283</td>
      <td>0.205378</td>
      <td>0.192142</td>
      <td>0.410210</td>
      <td>0.012785</td>
      <td>0.107436</td>
      <td>-0.171990</td>
      <td>...</td>
      <td>-0.002465</td>
      <td>-0.005723</td>
      <td>-0.002387</td>
      <td>0.001833</td>
      <td>-0.005137</td>
      <td>0.000737</td>
      <td>0.004796</td>
      <td>-0.003082</td>
      <td>-0.001402</td>
      <td>-0.002621</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.025882</td>
      <td>0.881859</td>
      <td>-0.031231</td>
      <td>0.003809</td>
      <td>-0.009610</td>
      <td>0.636919</td>
      <td>0.006099</td>
      <td>0.023052</td>
      <td>-0.019402</td>
      <td>0.196938</td>
      <td>...</td>
      <td>0.004825</td>
      <td>-0.002954</td>
      <td>0.002429</td>
      <td>0.024075</td>
      <td>-0.002344</td>
      <td>0.006869</td>
      <td>0.007860</td>
      <td>0.003550</td>
      <td>-0.000376</td>
      <td>0.004453</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.545908</td>
      <td>0.648594</td>
      <td>0.387437</td>
      <td>-0.008829</td>
      <td>0.219286</td>
      <td>0.852600</td>
      <td>0.037864</td>
      <td>0.083376</td>
      <td>0.211910</td>
      <td>0.977409</td>
      <td>...</td>
      <td>0.002507</td>
      <td>0.004520</td>
      <td>0.002546</td>
      <td>0.001082</td>
      <td>0.003604</td>
      <td>0.002208</td>
      <td>0.004599</td>
      <td>0.001259</td>
      <td>0.001489</td>
      <td>0.002720</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.023229</td>
      <td>1.073306</td>
      <td>1.197391</td>
      <td>0.106130</td>
      <td>1.185754</td>
      <td>0.646488</td>
      <td>1.362204</td>
      <td>0.203931</td>
      <td>0.284101</td>
      <td>1.453561</td>
      <td>...</td>
      <td>0.001165</td>
      <td>-0.001161</td>
      <td>-0.000224</td>
      <td>0.000626</td>
      <td>-0.000193</td>
      <td>-0.000454</td>
      <td>0.004080</td>
      <td>-0.002701</td>
      <td>0.000496</td>
      <td>0.000509</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 14365 columns</p>
</div>




```python
def recommend_movies(predictions_df, userId, movies_df, originalratings_df, num_recommendations):
    
    # Get and sort the user's predictions
    userrownumber = userId - 1 # userId starts at 1, not 0
    sortedpredictions = predicted_df.iloc[userrownumber].sort_values(ascending=False)
    
    # Get the user's data and merge in the movie information.
    userdata = originalratings_df[originalratings_df.userId == (userId)]
    usercomplete = (userdata.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False)
                 )

    print ('User {0} has already rated {1} movies.'.format(userId, usercomplete.shape[0]))
    print ('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['movieId'].isin(usercomplete['movieId'])].
         merge(pd.DataFrame(sortedpredictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns = {userrownumber: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return usercomplete, recommendations
```


```python
ratedalready, predictions = recommend_movies(predicted_df, 1003, movies_df, ratings_df, 10)
```

    User 1003 has already rated 174 movies.
    Recommending highest 10 predicted ratings movies not already rated.
    


```python
ratedalready.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>74</th>
      <td>1003</td>
      <td>2571</td>
      <td>5.0</td>
      <td>1209226214</td>
      <td>Matrix, The (1999)</td>
      <td>Action|Sci-Fi|Thriller</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1003</td>
      <td>1210</td>
      <td>5.0</td>
      <td>1209308048</td>
      <td>Star Wars: Episode VI - Return of the Jedi (1983)</td>
      <td>Action|Adventure|Sci-Fi</td>
    </tr>
    <tr>
      <th>134</th>
      <td>1003</td>
      <td>7153</td>
      <td>5.0</td>
      <td>1209226291</td>
      <td>Lord of the Rings: The Return of the King, The...</td>
      <td>Action|Adventure|Drama|Fantasy</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1003</td>
      <td>110</td>
      <td>5.0</td>
      <td>1209226276</td>
      <td>Braveheart (1995)</td>
      <td>Action|Drama|War</td>
    </tr>
    <tr>
      <th>130</th>
      <td>1003</td>
      <td>6874</td>
      <td>5.0</td>
      <td>1209226396</td>
      <td>Kill Bill: Vol. 1 (2003)</td>
      <td>Action|Crime|Thriller</td>
    </tr>
    <tr>
      <th>118</th>
      <td>1003</td>
      <td>5952</td>
      <td>5.0</td>
      <td>1209227126</td>
      <td>Lord of the Rings: The Two Towers, The (2002)</td>
      <td>Adventure|Fantasy</td>
    </tr>
    <tr>
      <th>94</th>
      <td>1003</td>
      <td>3578</td>
      <td>5.0</td>
      <td>1209226379</td>
      <td>Gladiator (2000)</td>
      <td>Action|Adventure|Drama</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1003</td>
      <td>1527</td>
      <td>4.5</td>
      <td>1209227148</td>
      <td>Fifth Element, The (1997)</td>
      <td>Action|Adventure|Comedy|Sci-Fi</td>
    </tr>
    <tr>
      <th>137</th>
      <td>1003</td>
      <td>7438</td>
      <td>4.5</td>
      <td>1209226530</td>
      <td>Kill Bill: Vol. 2 (2004)</td>
      <td>Action|Drama|Thriller</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1003</td>
      <td>780</td>
      <td>4.5</td>
      <td>1209226261</td>
      <td>Independence Day (a.k.a. ID4) (1996)</td>
      <td>Action|Adventure|Sci-Fi|Thriller</td>
    </tr>
  </tbody>
</table>
</div>




```python
predictions
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4759</th>
      <td>4963</td>
      <td>Ocean's Eleven (2001)</td>
      <td>Crime|Thriller</td>
    </tr>
    <tr>
      <th>5208</th>
      <td>5418</td>
      <td>Bourne Identity, The (2002)</td>
      <td>Action|Mystery|Thriller</td>
    </tr>
    <tr>
      <th>4788</th>
      <td>4993</td>
      <td>Lord of the Rings: The Fellowship of the Ring,...</td>
      <td>Adventure|Fantasy</td>
    </tr>
    <tr>
      <th>1605</th>
      <td>1721</td>
      <td>Titanic (1997)</td>
      <td>Drama|Romance</td>
    </tr>
    <tr>
      <th>2549</th>
      <td>2716</td>
      <td>Ghostbusters (a.k.a. Ghost Busters) (1984)</td>
      <td>Action|Comedy|Sci-Fi</td>
    </tr>
    <tr>
      <th>42</th>
      <td>47</td>
      <td>Seven (a.k.a. Se7en) (1995)</td>
      <td>Mystery|Thriller</td>
    </tr>
    <tr>
      <th>4692</th>
      <td>4896</td>
      <td>Harry Potter and the Sorcerer's Stone (a.k.a. ...</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>1220</th>
      <td>1291</td>
      <td>Indiana Jones and the Last Crusade (1989)</td>
      <td>Action|Adventure</td>
    </tr>
    <tr>
      <th>326</th>
      <td>344</td>
      <td>Ace Ventura: Pet Detective (1994)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>2690</th>
      <td>2858</td>
      <td>American Beauty (1999)</td>
      <td>Comedy|Drama</td>
    </tr>
  </tbody>
</table>
</div>



### **The recommendations look pretty solid!**

# Conclusion

Low-dimensional matrix recommenders try to capture the underlying features driving the raw data (which we understand as tastes and preferences). From a theoretical perspective, if we want to make recommendations based on people's tastes, this seems like the better approach. This technique also scales significantly better to larger datasets.

We do lose some meaningful signals by using a lower-rank matrix.

One particularly cool and effective strategy is to combine factorization and neighborhood methods into one framework(http://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf). This research field is extremely active, and you should check out this Coursera course, Introduction to Recommender Systems(https://www.coursera.org/specializations/recommender-systems), for understanding this better.
