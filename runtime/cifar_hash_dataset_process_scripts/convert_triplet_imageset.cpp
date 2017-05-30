// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_triplet_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME TRIPLET_NUMBER 
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of triplet files each line, in the format as
//   Pic1.jpg Pic2.jpg Pic3.jpg
//   ....


// added by Fuchen Long in 7.23.2016


#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/rng.hpp"

//port for win32
#ifdef _MSC_VER
#define snprintf sprintf_s
#endif

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
	"When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
	"Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "leveldb",
	"The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
	"When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
	"When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
	"Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
		"format used as input for Caffe.\n"
		"Usage:\n"
		"    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME TRIPLET_NUMBER\n"
		"The ImageNet dataset for the training demo is at\n"
		"    http://www.image-net.org/download-images\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 4) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
		return 1;
	}

	const bool is_color = !FLAGS_gray;
	const bool check_size = FLAGS_check_size;
	const bool encoded = FLAGS_encoded;
	const string encode_type = FLAGS_encode_type;

	std::ifstream infile(argv[2]);
	int NumberTriplet = atoi(argv[4]);
	std::vector<string> lines1;
	std::vector<string> lines2;
	std::vector<string> lines3;
	std::vector<string> filename;
	std::string filename1;
	std::string filename2;
	std::string filename3;
	int i = 0;
	for (int k = 0; k < NumberTriplet; k++) //change the number to fit your size
	{
		infile >> filename1;
		infile >> filename2;
		infile >> filename3;
		lines1.push_back(filename1); // original image
		lines2.push_back(filename2); // similar image
		lines3.push_back(filename3); //different image
		i++;
		if (i % 1000 == 0)
			LOG(INFO) << "have load " << i << " triplet lines.\n";
	}
	LOG(INFO) << "Start to make triplet DB." << std::endl;
	LOG(INFO) << "A total of " << lines1.size() << " images.";
	if (encode_type.size() && !encoded)
		LOG(INFO) << "encode_type specified, assuming encoded=true.";

	int resize_height = std::max<int>(0, FLAGS_resize_height);
	int resize_width = std::max<int>(0, FLAGS_resize_width);
	if (resize_height > 0 && resize_width > 0)
	{
		LOG(INFO) << "Resize images: resize_width=" << resize_width << ",resize_height=" << resize_height;
	}
	else
	{
		LOG(INFO) << "No resize images.";
	}

	// Create new DB
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[3], db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	// Storing to db
	std::string root_folder(argv[1]);
	Datum datum;
	int count = 0;
	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];
	int data_size = 0;
	bool data_size_initialized = false;

	for (int line_id = 0; line_id < lines1.size(); ++line_id) {
		bool status;
		status = MultiImageToData(root_folder + lines1[line_id],
			root_folder + lines2[line_id],
			root_folder + lines3[line_id],
			resize_height, resize_width,
			is_color, &datum);
		if (status == false) continue;
		if (check_size) {
			if (!data_size_initialized) {
				data_size = datum.channels() * datum.height() * datum.width(); //the three channels
				data_size_initialized = true;
			}
			else {
				const std::string& data = datum.data();
				CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
					<< data.size();
			}
		}
		// sequential
		int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
			lines1[line_id].c_str());

		// Put in db
		string out;
		CHECK(datum.SerializeToString(&out));
		txn->Put(string(key_cstr, length), out);

		if (++count % 1000 == 0) {
			// Commit db
			txn->Commit();
			txn.reset(db->NewTransaction());
			LOG(ERROR) << "Processed " << count << " files.\n";
			//std::cout << "Processed " << count << " files\n";
		}
	}
	// write the last batch
	if (count % 1000 != 0) {
		txn->Commit();
		LOG(ERROR) << "Processed " << count << " files.";
	}
	return 0;
}