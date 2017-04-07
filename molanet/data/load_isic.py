import json
import sys

import requests
from bson import Binary

from molanet.data.database import MongoConnection
from molanet.data.entities import MoleSample, Diagnosis, Segmentation, SkillLevel

segmentationurl = r"https://isic-archive.com:443/api/v1/segmentation"
segmentationParams = {"limit": "50", "offset": "0", "sort": "created", "sortdir": "-1"}

imagesListUrl = r"https://isic-archive.com:443/api/v1/image"
imagesListParams = {"limit": "50", "offset": "0", "sort": "name", "sortdir": "1"}
imageDetailsUrl = "https://isic-archive.com:443/api/v1/image/"

def addParam(param, key, value):
    param[key] = str(value)
    return param

def segmentationParamFromID(id):
    return addParam(segmentationParams, "imageId", id)


def list_images(numberOfImages):
    r = requests.get(url=imagesListUrl, params=addParam(imagesListParams, "limit", numberOfImages))
    data = json.loads(r.text)
    return data

def getImagesDetails(id):
    r = requests.get(url=imageDetailsUrl + id)
    return json.loads(r.text)

def getSegmentationInfo(id):
    r = requests.get(url=segmentationurl, params=segmentationParamFromID(id))
    return json.loads(r.text)


_uuid = 'uuid'
_name = 'name'  # eg ISIC_00001
_idInDataSet = 'idInDataset'  # eg 5436e3acbae478396759f0d1
_dataSource = 'dataSource'
_datasetName = 'datasetName'
_dimensionsX = 'dimensionsX'
_dimensionsY = 'dimensionsY'
_benign_malignant = 'benign_malignant'
_diagnosis = 'diagnosis'
_imageDownloadURl = 'imageDownloadURl'
_imageSuperPixelsDownloadUrl = 'imageSuperPixelsDownloadUrl'
_dataSetUniqueID = 'dataSetUniqueID'
_segmentation = 'segmentation'


def getImageInfo(id, resultDict={}, debug=False):
    data = getImagesDetails(id)
    resultDict[_dataSource] = 'isic'
    resultDict[_uuid] = resultDict[_dataSource] + id
    resultDict[_datasetName] = data['dataset']['name']
    resultDict[_dataSetUniqueID] = data['dataset']['_id']
    resultDict[_idInDataSet] = id
    resultDict[_name] = data['name']
    resultDict[_dimensionsX] = data['meta']['acquisition']['pixelsX']
    resultDict[_dimensionsY] = data['meta']['acquisition']['pixelsY']
    resultDict[_benign_malignant] = data['meta']['clinical']['benign_malignant']
    resultDict[_diagnosis] = data['meta']['clinical']['diagnosis']
    resultDict[_imageDownloadURl] = 'https://isic-archive.com:443/api/v1/image/' + id + '/download'
    resultDict[_imageSuperPixelsDownloadUrl] = 'https://isic-archive.com:443/api/v1/image/' + id + '/superpixels'


    segmentations = getSegmentationInfo(id)
    resultDict[_segmentation] = segmentations
    for seg in segmentations:
        seg['segmentationDownloadUrl'] = \
            'https://isic-archive.com:443/api/v1/segmentation/' + seg['_id'] + '/mask?contentDisposition=inline'

    if debug:
        print(resultDict)
        # save stuff to file or something
        f = open('segmentation' + resultDict[_segmentation][0]['_id'] + '.jpg', 'wb')
        f.write(requests.get(resultDict[_segmentation][0]['segmentationDownloadUrl']).content)
        f.close()
        # save to image to a file or something
        f = open(resultDict[_name] + '.jpg', 'wb')
        f.write(requests.get(resultDict[_imageDownloadURl]).content)
        f.close()
        f = open(resultDict[_name] + '_data.json', 'w')
        f.write(json.dumps(resultDict))
        f.close()

    return resultDict


def connect(url, port, user, pw, dbname):
    dbconnection = MongoConnection(user, pw, url, dbname)
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


def parse_segmentation(segmentations, dimensions):
    def parseSkill(skill):
        switch = {
            'novice': SkillLevel.NOVICE,
            'expert': SkillLevel.EXPERT
        }
        return switch[skill]
    parsed = []
    for seg in segmentations:
        parsed.append(Segmentation(seg['_id'],
                                   Binary(requests.get(seg['segmentationDownloadUrl']).content),
                                   parseSkill(seg['skill']),
                                   dimensions))
    return parsed


def parsedata(dict):
    dim = (dict[_dimensionsX], dict[_dimensionsY])
    return MoleSample(dict[_uuid],
                      dict[_dataSource],
                      dict[_datasetName],
                      dict[_idInDataSet],
                      dict[_name],
                      dim,
                      Diagnosis(parse_diagnosis(dict[_benign_malignant])),
                      Binary(requests.get(dict[_imageDownloadURl]).content),
                      parse_segmentation(dict[_segmentation], dimensions=None))


def load_isic(maxImages=15000):
    dbconnection = parsedbconnectioninfo_connect('dbconnection.json')
    # dbconnection = MongoConnection(url='mongodb://localhost:27017/', db_name='molanet')
    data = list_images(maxImages)
    count = 0
    for imageData in data:
        sample = parsedata(getImageInfo(imageData['_id']))
        from pymongo.errors import BulkWriteError
        try:
            dbconnection.load_data_set([sample])
        except BulkWriteError:
            exc_info = sys.exc_info()
            print(exc_info)
            print("probably attempted to overwrite existing document _id=%s (aka. %s)" % (sample.uuid, sample.name))
            pass
        # dosomething with it
        print("%d : %s from dataset %s done" % (count, sample.name, sample.data_set))
        count += 1


# p = parsedata(getImageInfo("5436e3abbae478396759f0cf",debug=True))
# load_isic(maxImages=5)
load_isic()
