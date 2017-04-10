import io
import json
import sys
import os

import numpy as np
import requests
from PIL import Image
from bson.binary import Binary
from pymongo.errors import DocumentTooLarge

from molanet.data.database import MongoConnection
from molanet.data.entities import MoleSample, Diagnosis, Segmentation, SkillLevel

segmentationurl = r"https://isic-archive.com:443/api/v1/segmentation"
segmentationParams = {"limit": "50", "offset": "0", "sort": "created", "sortdir": "-1"}

imagesListUrl = r"https://isic-archive.com:443/api/v1/image"
imagesListParams = {"limit": "50", "offset": "0", "sort": "name", "sortdir": "1"}
imageDetailsUrl = "https://isic-archive.com:443/api/v1/image/"


def addparam(param, key, value):
    param[key] = str(value)
    return param


def list_images(maxnumberofimages: int, offset: int):
    p = addparam(imagesListParams, "limit", maxnumberofimages)
    r = requests.get(url=imagesListUrl, params=addparam(p, 'offset', str(offset)))
    data = json.loads(r.text)
    return data


def getimagedetails(id):
    r = requests.get(url=imageDetailsUrl + id)
    return json.loads(r.text)


def getsegmentationinfo(id):
    r = requests.get(url=segmentationurl, params=addparam(segmentationParams, "imageId", id))
    return json.loads(r.text)


def downloadbinary_isic_image(id):
    url = 'https://isic-archive.com:443/api/v1/image/' + id + '/download'
    return requests.get(url).content


def downloadbinary_isic_superpixels(id):
    url = 'https://isic-archive.com:443/api/v1/image/' + id + '/superpixels'
    return requests.get(url).content


def downloadbinary_isic_segmentation(segmentation_id):
    url = 'https://isic-archive.com:443/api/v1/segmentation/' + segmentation_id + '/mask?contentDisposition=inline'
    return requests.get(url).content


def convert_binaryimage_numpyarray(im_binary: bytes):
    image = Image.open(io.BytesIO(im_binary))
    return np.array(image)


_uuid = 'uuid'
_name = 'name'  # eg ISIC_00001
_idInDataSet = 'idInDataset'  # eg 5436e3acbae478396759f0d1
_dataSource = 'dataSource'
_datasetName = 'datasetName'
_dimensionsX = 'dimensionsX'
_dimensionsY = 'dimensionsY'
_benign_malignant = 'benign_malignant'
_diagnosis = 'diagnosis'
_imageSuperPixelsDownloadUrl = 'imageSuperPixelsDownloadUrl'
_dataSetUniqueID = 'dataSetUniqueID'
_segmentation = 'segmentation'


def getimageinfo(im_id, debug=False):
    resultdict = {}
    data = getimagedetails(im_id)
    resultdict[_dataSource] = 'isic'
    resultdict[_uuid] = resultdict[_dataSource] + im_id
    resultdict[_datasetName] = data['dataset']['name']
    resultdict[_dataSetUniqueID] = data['dataset']['_id']
    resultdict[_idInDataSet] = im_id
    resultdict[_name] = data['name']
    resultdict[_dimensionsX] = data['meta']['acquisition']['pixelsX']
    resultdict[_dimensionsY] = data['meta']['acquisition']['pixelsY']
    resultdict[_benign_malignant] = data['meta']['clinical']['benign_malignant']
    resultdict[_diagnosis] = data['meta']['clinical']['diagnosis']

    segmentations = getsegmentationinfo(im_id)
    resultdict[_segmentation] = segmentations
    if debug:
        print(resultdict)
        # save stuff to file or something
        f = open('segmentation' + resultdict[_segmentation][0]['_id'] + '.jpg', 'wb')
        f.write(downloadbinary_isic_segmentation(resultdict[_segmentation][0]['_id']))
        f.close()
        # save to image to a file or something
        f = open(resultdict[_name] + '.jpg', 'wb')
        f.write(downloadbinary_isic_image(resultdict[_idInDataSet]))
        f.close()
        f = open(resultdict[_name] + '_data.json', 'w')
        f.write(json.dumps(resultdict))
        f.close()

    return resultdict


def connect(url, port, user, pw, dbname):
    dbconnection = MongoConnection(url=url, username=user, password=pw, db_name=dbname)
    return dbconnection


def parsedbconnectioninfo_connect(file):
    f = open(file, 'r')
    params = json.loads(f.read())
    return connect(params['url'], params['port'], params['user'], params['pass'], 'molanet')


def parse_diagnosis(diagnosis):
    switch = {
        'benign': Diagnosis.BENIGN,
        'malignant': Diagnosis.MALIGNANT,
    }
    return switch.get(diagnosis, Diagnosis.UNKNOWN)


def parse_segmentation(segmentations, dimensions: (int, int)):
    def parseskill(skill):
        switch = {
            'novice': SkillLevel.NOVICE,
            'expert': SkillLevel.EXPERT
        }
        return switch[skill]

    parsed = []
    images = {}
    for seg in segmentations:
        segmentation_image = downloadbinary_isic_segmentation(seg['_id'])
        segmentation_np_image = convert_binaryimage_numpyarray(segmentation_image)
        parsed.append(Segmentation(seg['_id'],
                                   Binary(np.ndarray.tobytes(segmentation_np_image)),
                                   parseskill(seg['skill']),
                                   dimensions))
        images[seg['_id']] = (segmentation_image, segmentation_np_image)
    return parsed, images


def parsedata(data):
    dim = (data[_dimensionsX], data[_dimensionsY])
    image = downloadbinary_isic_image('5436e3abbae478396759f0cf')
    np_image = convert_binaryimage_numpyarray(image)
    (segmentations, segmentations_raw) = parse_segmentation(data[_segmentation], dimensions=None)
    return (MoleSample(data[_uuid],
                       data[_dataSource],
                       data[_datasetName],
                       data[_idInDataSet],
                       data[_name],
                       dim,
                       Diagnosis(parse_diagnosis(data[_benign_malignant])),
                       Binary(np.ndarray.tobytes(np_image)),
                       segmentations), (image, np_image), segmentations_raw)


def load_isic(maximages=15000, offset=0, logFile=None, images_dir=None, npimagesDir=None, seg_dir=None,
              np_seg_dir=None):
    dbconnection = parsedbconnectioninfo_connect('dbconnection.json')
    # dbconnection = MongoConnection(url='mongodb://localhost:27017/', db_name='molanet')
    data = list_images(maximages, offset)
    count = offset
    for imageData in data:
        sample, (image, np_image), seg_raw = parsedata(getimageinfo(imageData['_id']))

        if (images_dir is not None):
            with open(os.path.join(images_dir, sample.uuid + '.jpg'), 'wb') as f:
                f.write(image)
        if (npimagesDir is not None):
            np.save(os.path.join(npimagesDir, sample.uuid), np_image)
        if (seg_dir is not None and np_seg_dir is not None):
            for seg_id, (seg, np_seg) in seg_raw.items():
                with open(os.path.join(seg_dir, sample.uuid + seg_id + '.jpg'), 'wb') as f:
                    f.write(seg)
                np.save(os.path.join(np_seg_dir, sample.uuid + seg_id), np_seg)

        from pymongo.errors import BulkWriteError
        try:
            dbconnection.load_data_set([sample])
        except BulkWriteError:
            exc_info = sys.exc_info()
            print(exc_info)
            print("probably attempted to overwrite existing document _id=%s (aka. %s)" % (sample.uuid, sample.name))
            pass
        except DocumentTooLarge:
            exc_info = sys.exc_info()
            print(exc_info)
            print("too large document _id=%s (aka. %s)" % (sample.uuid, sample.name))
            if logFile is not None:
                logFile.write('%d Document to large _id=%s (aka. %s)' % (count, sample.uuid, sample.name))
                logFile.flush()
            pass
        # dosomething with it
        print("%d : %s from dataset %s done" % (count, sample.name, sample.data_set))
        count += 1


# p = parsedata(getImageInfo("5436e3abbae478396759f0cf",debug=True))
# im = convert_binaryimage_numpyarray(downloadbinary_isic_image('5436e3abbae478396759f0cf'))
# print(im.shape)
# seg = convert_binaryimage_numpyarray(downloadbinary_isic_segmentation('5463934bbae47821f88025ad'))
# print(seg.shape)
# load_isic(maxImages=5)
def mk_dirs_if_not_exists(rootfolder_path='samples'):
    images = os.path.join(rootfolder_path, 'images')
    segmentations = os.path.join(rootfolder_path, 'segmentations')
    images_numpy = os.path.join(rootfolder_path, 'images_numpy')
    segmentations_numpy = os.path.join(rootfolder_path, 'segmentations_numpy')

    if not os.path.exists(rootfolder_path):
        os.makedirs(rootfolder_path)
    if not os.path.exists(images):
        os.makedirs(images)
    if not os.path.exists(segmentations):
        os.makedirs(segmentations)
    if not os.path.exists(images_numpy):
        os.makedirs(images_numpy)
    if not os.path.exists(segmentations_numpy):
        os.makedirs(segmentations_numpy)

    return (images, segmentations, images_numpy, segmentations_numpy)


(images, segmentations, images_numpy, segmentations_numpy) = mk_dirs_if_not_exists('samples')
logFilePath = "log_load_isic.txt"
with open(logFilePath, 'w') as log:
    load_isic(offset=377, logFile=log, images_dir=images, npimagesDir=images_numpy, seg_dir=segmentations,
              np_seg_dir=segmentations_numpy)

mk_dirs_if_not_exists()
