/*
* Description: 
* - Implementation of metric of mean average precision(MAP) 
*   in retrieval task.
* - Find the visual similar images of each image in the 
*   query set from the pool image set.
* - Given a query image, the retrieval list of images is produced
*   by sorting the hamming distances between the query image
*   and images in search pool.
* 
* Copyright: HYPJUDY, 2017.4.12
* 
*/
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
using namespace std;

// calculate MAP with the top x returned neighbors
const int top_neighbor_num = 10000; // if query in whole pool: pools.size()

struct Image {
	string hashcode;
    unsigned int label, dist;
	Image(string _hashcode, unsigned int _label = 1,
     unsigned int _dist = 0) 
	: hashcode(_hashcode), label(_label), dist(_dist) {}
};

struct MAP_LABEL {
  float map;
  int num;
  MAP_LABEL(float _m = 0, int _n = 0) : map(_m), num(_n) {}
};

/* The Hamming distance between two strings of 
*  equal length is the number of positions at 
*  which the corresponding symbols are different.
*/
int hamming_distance(string x, string y) {
    int dist = 0;
    int i = x.size() - 1, j = y.size() - 1;
    for(i, j; i >= 0 && j >= 0; --i, --j) { 
        if(x[i] != y[i]) ++dist;
    }
    while(i != -1) {
      if(x[i--]) ++dist;
    }
    while(j != -1) {
      if(y[j--]) ++dist;
    }
    return dist;
}

/* Sort by hamming distance */
int cmp(Image x, Image y) {
	return x.dist < y.dist;
}

int main() {
  float MAP = 0.0; // mean average precision
  struct MAP_LABEL MAP_labels[11]; // MAP of each label

  const char * query_path = "cifar_hash_dataset/query_hashcode.txt";
  const char * query_label_path = "cifar_hash_dataset/query_label.txt";
  const char * pool_path = "cifar_hash_dataset/pool_hashcode.txt";
  const char * pool_label_path = "cifar_hash_dataset/pool_label.txt";
  cout << "Reading images's hashcode from files..." << endl;

  // read from image's pool, store hashcode
  ifstream infile_pool; 
  infile_pool.open(pool_path);
  vector<Image> pools; // one pool, 's' indicates images
  string hashcode;
  unsigned int label;
  while (infile_pool >> hashcode) {
  	Image img(hashcode);
  	pools.push_back(img);
  }

  // store pool labels in order
  ifstream infile_pool_label; 
  infile_pool_label.open(pool_label_path);
  unsigned int i = 0;
  while (infile_pool_label >> label) {
  	if(i < pools.size()) pools[i++].label = label;
  }

  // store query labels in order
  ifstream infile_query_label; 
  infile_query_label.open(query_label_path);
  vector<unsigned int> query_labels;
  while (infile_query_label >> label) {
  	query_labels.push_back(label);
  }
  
  cout << "Starting querying images in " \
    << pools.size() << " images' pool... " << endl;

  // read query image's hashcode and calculate dist
  ifstream infile_query; 
  infile_query.open(query_path);
  unsigned idx_query = 0;
  while (infile_query >> hashcode) { // for each query image
  	// calculate its dist between each image in pool
  	for(int i = 0; i < pools.size(); ++i)
  		pools[i].dist = hamming_distance(pools[i].hashcode, hashcode);
  	// sort, smallest hamming distance rank fist
  	sort(pools.begin(), pools.end(), cmp);
   	unsigned int query_label = query_labels[idx_query++];
    cout << "\n\n-------- query hashcode#" << idx_query + 1 << ": " << hashcode \
         << " ,label " << query_label << "--------" << endl;
  	// iterate images in pool, consider each similar image(same label)
    float MAP_sub = 0.0; // for each query image
    int num = 1;
    // Since the cifar10 dataset I used ordered by label 1 to 10
    // every 10 images, the number of total similar/relevant
    // images (same label) is pool image set's size divided by 10:
    int denominator = min(static_cast<int>(pools.size() / 10), top_neighbor_num);
    MAP_labels[query_label].num += 1;
    for(int idx_pool = 0; idx_pool < top_neighbor_num && 
                          idx_pool < pools.size(); ++idx_pool) {
        // cout << "pool hashcode#" << idx_pool << ": " \
        //      << pools[idx_pool].hashcode \
        //      << ", label " << pools[idx_pool].label \
        //      << ", dist " << pools[idx_pool].dist << endl;
    	if(pools[idx_pool].label == query_label) { 
            // find an image with the same label in pool
            float tmp = 1.0 * num++ / (idx_pool + 1);
            MAP_sub += tmp;
            MAP_labels[query_label].map += (tmp / denominator);
            cout << "*** Find similar image#" << num - 1 
                 << "! Current query#" << idx_query + 1 \
                 << " precision is: " << MAP_sub << " ***\n" << endl;
    	}
    }
    if(denominator) MAP_sub /= denominator;
    else MAP_sub = 0; // avoid zero denominator
    
    MAP += MAP_sub;
    cout << "\nmAP of query image#" << idx_query + 1 << ": " << MAP_sub << endl;
  }
  MAP /= (idx_query + 1);
  cout << "Finish querying " << idx_query + 1 << " images in " 
       << pools.size() << " pool images set within the top " 
       << top_neighbor_num << " returned neighbors!" << endl;
  cout << "Mean Average Precision(MAP): " << MAP << endl;
  
  for(int i = 1; i <= 10; ++i) {
    if(MAP_labels[i].num)
      MAP_labels[i].map /= MAP_labels[i].num;
    else
      MAP_labels[i].map = 0;
    cout << "MAP of label#" << i << ": " << MAP_labels[i].map << endl;
  }
  
  
  infile_pool.close();
  infile_pool_label.close();
  infile_query_label.close();
  infile_query.close();
  return 0;
}
