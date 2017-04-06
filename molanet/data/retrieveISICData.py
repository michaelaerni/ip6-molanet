import requests
import json

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

def getImages(numberOfImages):
    r = requests.get(url=imagesListUrl, params=addParam(imagesListParams, "limit", numberOfImages))
    data = json.loads(r.text)
    return data

def getImagesDetails(id):
    r = requests.get(url=imageDetailsUrl + id)
    print(r.text)
    return json.loads(r.text)

def getSegmentationInfo(id):
    r = requests.get(url=segmentationurl, params=segmentationParamFromID(id))
    return json.loads(r.text)

def getImageInfo(id, resultDict={}):
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

    data = getImagesDetails(id)
    resultDict[_uuid] = 'ISIC' + id
    resultDict[_dataSource] = 'ISIC'
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

    print(resultDict)
    # save stuff to file or something
    f = open('segmentation.jpg', 'wb')
    f.write(requests.get(resultDict[_segmentation][0]['segmentationDownloadUrl']).content)
    f.close()

    # save to image to a file or something
    f = open('00000001.jpg', 'wb')
    f.write(requests.get(resultDict[_imageDownloadURl]).content)
    f.close()

    f = open('data', 'w')
    f.write(json.dumps(resultDict))
    f.close()
    return resultDict


def scrape():
    maxImages = 15000
    data = getImages(maxImages)
    count = 0
    for imageData in data.items:
        details = getImageInfo(imageData['_id'])
        # dosomething with it
        count += 1
        if (count % 200 == 0):
            print('retrieved ' + count + ' images')


getImageInfo("5436e3acbae478396759f0d7")
