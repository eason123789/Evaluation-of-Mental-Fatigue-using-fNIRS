

import functools
import json
import os
import pathlib
import re
import shutil
from datetime import datetime
from os import (makedirs,rename,rmdir,remove)
from os.path import (join,exists,isfile)
from pathlib import Path
from re import (findall)
from shutil import unpack_archive

import requests
from joblib import Parallel,delayed
from tqdm import tqdm
from tqdm.auto import tqdm

from old_code.gtheory.utils.misc import split_id_session

# pt = cf.root_path ()

def remove_temp_zip_file(source_folder):
  destination_folder,f=os.path.split(source_folder)
  new_path=os.path.join(destination_folder,f'{f}_tempt')
  os.rename(source_folder, new_path)

  for file_name in os.listdir(new_path):
      # construct full file path
      source = os.path.join(new_path,file_name)
      destination = os.path.join(destination_folder,file_name)
  
      # move only files
      if os.path.isfile(source):
          shutil.move(source, destination)
          # print('Moved:', file_name)
  rmdir( new_path )
  # mm=1

def split_id_session_url(myurl) :
  '''
    :param fpath:
    :return:
    '''
  
  sbj=findall(r's\d{2}',myurl,flags=re.IGNORECASE)[0].upper()
  sub_session=findall(r'\d{6}',myurl)[0]
  
  try:
    session_repeat=re.search(fr'{sub_session}(.*?).set',myurl).group(1)
  except:
    print('NA')
    session_repeat=''
  
  
  return sbj,sub_session,session_repeat

def download_p(url, filename):
  
  
  r = requests.get(url, stream=True, allow_redirects=True)
  if r.status_code != 200:
    r.raise_for_status()  # Will only raise for 4xx codes, so...
    raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
  file_size = int(r.headers.get('Content-Length', 0))
  
  path = pathlib.Path(filename).expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  
  desc = "(Unknown total file size)" if file_size == 0 else ""
  r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
  with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
    with path.open("wb") as f:
      shutil.copyfileobj(r_raw, f)
  
  return path


def rename_folder (root_dir, sbj, sub_session, file_path,file_path_temp,fname):

    file_new = join ( root_dir, sbj, sub_session )

    try:
        rename ( file_path_temp , file_new )
    except OSError:
        raise ('duplicate folder name')
        # raise Warning ('Potentially a duplicated file is there')

def filter_path_list(path_all):

    new_list=[]
    for xpath in path_all:
        sbj= findall ( r'S\d{2}', xpath,flags=re.IGNORECASE)
        if bool(sbj):
            new_list.append(xpath)
    return new_list
def get_extract_remove_zip (path_zip, root_dir, remove_zip):

        sbj, sub_session,root_folder = split_id_session ( path_zip )
        fpath,_ = os.path.splitext(path_zip)
        fname = (os.path.basename(fpath))

        file_path = join ( root_dir, sbj )
        file_path_store = join ( root_dir, sbj,sub_session )
        file_path_temp = join ( root_dir, sbj,fname )
        if not exists ( file_path_store ):
            makedirs ( file_path_store )  # create folder if it does not exist

        if isfile(file_path_store):
            # This to avoid extracting file that already exist
            if remove_zip: remove ( path_zip )  # Remove the zip file to conserve storage
            return
        # path_all = glob ( f'{file_path}/*')

        unpack_archive ( path_zip, file_path )
        # ZipFile(path_zip).extractall(file_path)

        # TODO Issue with subject S14, the .set folder not deleted properly
        if remove_zip: remove ( path_zip )  # Remove the zip file to conserve storage
        rename_folder ( root_dir, sbj, sub_session, file_path,file_path_temp,fname )

def extract_remove_zip (path_all, root_dir, remove_zip=False):
    path_all=filter_path_list(path_all)
    njob=1
    if njob==1:
        for path_zip in tqdm ( path_all, desc='Extracting zip file' ):
            get_extract_remove_zip(path_zip,root_dir, remove_zip)
    else:
        print('Using parallel to extract zip file')
        Parallel ( n_jobs=-1,verbose=50 ) ( delayed ( get_extract_remove_zip ) ( path_zip,root_dir, remove_zip )
                                     for path_zip in path_all)




# extract_remove_zip()

def download (tkey,d, dest_folder):

    sbj_detail=tkey
    turl=d[tkey][0]
    subject_id,sub_session,session_repeat=split_id_session_url(sbj_detail)
    file_path = join ( dest_folder, subject_id,f'{sub_session}{session_repeat}' )
    # file_path = join ( dest_folder, subject_id )
    if not Path ( file_path ).exists ():  # works for both file and folders

        if not exists ( file_path ):
            makedirs ( file_path )  # create folder if it does not exist


    npath=os.path.join(file_path,f'{sub_session}{session_repeat}_temp.zip')

    dir_root=download_p(turl, npath)
    unpack_archive ( dir_root,file_path)
    remove ( dir_root )
    fname=tkey
    new_path=os.path.join(file_path,fname)
    remove_temp_zip_file(new_path)


def extract_dic(data):
      alist=[[i['name'], i['downloadUrl']] for i in data['data']['publicItem']['files']['items']]
      return alist

def _get_file_web ():


    limit = 40
    all_data=[]
    for noffset in [0,40]:

      offset=noffset
      data = requests.post('https://figshare.com/api/graphql?thirdPartyCookies=true&type=current&operation=getPublicItemFiles',
                   json = {"operationName":"getPublicItemFiles","variables":{"itemId":7666055,"version":3,"offset":offset,"limit":limit},"query":"query getPublicItemFiles($itemId: Int!, $version: Int, $offset: Int!, $limit: Int!) {\n  publicItem: itemVersion(id: $itemId, version: $version) {\n    id\n    files(offset: $offset, limit: $limit) {\n      hasMore\n      items: elements {\n        id\n        name\n        status\n        extension\n        size\n        viewerType\n        mimeType\n        virusScanInfo {\n          virusFound\n        }\n        md5\n        isLinkOnly\n        thumb\n        previewMeta\n        suppliedMd5\n        previewState\n        previewLocation\n        downloadUrl\n      }\n    }\n  }\n}\n"}).json()

      all_data.append(data)


    files = [extract_dic(ndata) for ndata in all_data]
    flat_list = [item for sublist in files for item in sublist]

    alist_download=[]
    for nlist in flat_list:
      x = re.search("^.*.set.zip$", nlist[0])
      if x:
        alist_download.append(nlist)
    # alist_download=[['s01_051017m.set.zip', 'https://figshare.com/ndownloader/files/14242478'], ['s01_060227n.set.zip', 'https://figshare.com/ndownloader/files/14249780'], ['s01_060926_1n.set.zip', 'https://figshare.com/ndownloader/files/14249783'], ['s01_060926_2n.set.zip', 'https://figshare.com/ndownloader/files/14249786'], ['s01_061102n.set.zip', 'https://figshare.com/ndownloader/files/14249789'], ['s02_050921m.set.zip', 'https://figshare.com/ndownloader/files/14249792'], ['s02_051115m.set.zip', 'https://figshare.com/ndownloader/files/14249795'], ['s04_051130m.set.zip', 'https://figshare.com/ndownloader/files/14249798'], ['s05_051120m.set.zip', 'https://figshare.com/ndownloader/files/14249801'], ['s05_060308n.set.zip', 'https://figshare.com/ndownloader/files/14249804'], ['s05_061019m.set.zip', 'https://figshare.com/ndownloader/files/14249807'], ['s05_061101n.set.zip', 'https://figshare.com/ndownloader/files/14249810'], ['s06_051119m.set.zip', 'https://figshare.com/ndownloader/files/14249849'], ['s09_060313n.set.zip', 'https://figshare.com/ndownloader/files/14249852'], ['s09_060317n.set.zip', 'https://figshare.com/ndownloader/files/14249855'], ['s09_060720_1n.set.zip', 'https://figshare.com/ndownloader/files/14249858'], ['s11_060920_1n.set.zip', 'https://figshare.com/ndownloader/files/14249867'], ['s12_060710_1m.set.zip', 'https://figshare.com/ndownloader/files/14249873'], ['s12_060710_2m.set.zip', 'https://figshare.com/ndownloader/files/14249876'], ['s13_060213m.set.zip', 'https://figshare.com/ndownloader/files/14249879'], ['s13_060217m.set.zip', 'https://figshare.com/ndownloader/files/14249882'], ['s14_060319m.set.zip', 'https://figshare.com/ndownloader/files/14249885'], ['s14_060319n.set.zip', 'https://figshare.com/ndownloader/files/14249888'], ['s22_080513m.set.zip', 'https://figshare.com/ndownloader/files/14249891'], ['s22_090825n.set.zip', 'https://figshare.com/ndownloader/files/14249894'], ['s22_090922m.set.zip', 'https://figshare.com/ndownloader/files/14249900'], ['s22_091006m.set.zip', 'https://figshare.com/ndownloader/files/14249903'], ['s23_060711_1m.set.zip', 'https://figshare.com/ndownloader/files/14249906'], ['s31_061020m.set.zip', 'https://figshare.com/ndownloader/files/14249909'], ['s31_061103n.set.zip', 'https://figshare.com/ndownloader/files/14249912'], ['s35_070115m.set.zip', 'https://figshare.com/ndownloader/files/14249915'], ['s35_070322n.set.zip', 'https://figshare.com/ndownloader/files/14249933'], ['s40_070124n.set.zip', 'https://figshare.com/ndownloader/files/14249945'], ['s40_070131m.set.zip', 'https://figshare.com/ndownloader/files/14249948'], ['s41_061225n.set.zip', 'https://figshare.com/ndownloader/files/14249951'], ['s41_080520m.set.zip', 'https://figshare.com/ndownloader/files/14249957'], ['s41_080530n.set.zip', 'https://figshare.com/ndownloader/files/14249960'], ['s41_090813m.set.zip', 'https://figshare.com/ndownloader/files/14250044'], ['s41_091104n.set.zip', 'https://figshare.com/ndownloader/files/14250119'], ['s42_061229n.set.zip', 'https://figshare.com/ndownloader/files/14250161'], ['s42_070105n.set.zip', 'https://figshare.com/ndownloader/files/14250245'], ['s43_070202m.set.zip', 'https://figshare.com/ndownloader/files/14250320'], ['s43_070205n.set.zip', 'https://figshare.com/ndownloader/files/14250368'], ['s43_070208n.set.zip', 'https://figshare.com/ndownloader/files/14250401'], ['s44_070126m.set.zip', 'https://figshare.com/ndownloader/files/14250404'], ['s44_070205n.set.zip', 'https://figshare.com/ndownloader/files/14250413'], ['s44_070209m.set.zip', 'https://figshare.com/ndownloader/files/14250419'], ['s44_070325n.set.zip', 'https://figshare.com/ndownloader/files/14250437'], ['s45_070307n.set.zip', 'https://figshare.com/ndownloader/files/14250440'], ['s45_070321n.set.zip', 'https://figshare.com/ndownloader/files/14250443'], ['s48_080501n.set.zip', 'https://figshare.com/ndownloader/files/14250449'], ['s49_080522n.set.zip', 'https://figshare.com/ndownloader/files/14250452'], ['s49_080527n.set.zip', 'https://figshare.com/ndownloader/files/14250458'], ['s49_080602m.set.zip', 'https://figshare.com/ndownloader/files/14250461'], ['s50_080725n.set.zip', 'https://figshare.com/ndownloader/files/14250464'], ['s50_080731m.set.zip', 'https://figshare.com/ndownloader/files/14250467'], ['s52_081017n.set.zip', 'https://figshare.com/ndownloader/files/14250473'], ['s53_081018n.set.zip', 'https://figshare.com/ndownloader/files/14250476'], ['s53_090918n.set.zip', 'https://figshare.com/ndownloader/files/14250479'], ['s53_090925m.set.zip', 'https://figshare.com/ndownloader/files/14250494'], ['s54_081226m.set.zip', 'https://figshare.com/ndownloader/files/14250497'], ['s55_090930n.set.zip', 'https://figshare.com/ndownloader/files/14250500']]


    _download ( alist_download )



def check_if_file_available(d,npath,overwrite=False):
    import os
    if overwrite is False:
        if os.path.isfile(npath):
            f=open(npath,"r")
            d_prev=json.loads(f.read())
            alist_to_download=[ndict for ndict in d_prev if d_prev[ndict][1]!='Available Locally']
          ### Need to check what has succesfully download before
    else:
        with open(npath,'w') as fp :
            json.dump(d,fp)
        alist_to_download=d
    
    return alist_to_download

def _progress(npath,tkey,tstatus):
    now=datetime.now()
    f=open(npath,"r")
    d=json.loads(f.read())
    
    d[tkey][1]=tstatus
    d[tkey][2]=now.strftime("%d_%m_%Y_%H_%M_%S")
    
    with open(npath,'w') as fp :
            json.dump(d,fp)
      
def _download (url_ls):
    '''
    For some reason, parallel x working. SO we only use series
    TODO Reporting for success donwload
    '''
    from old_code.gtheory.utils.misc import check_make_folder
    persons_dict = {x[0].split('.zip')[0]: [*x[1:],'Not Available Locally','ntime'] for x in url_ls}
    nfile='prog_download.json'
    check_make_folder(pt['report'],remove=False)
    preport=os.path.join(pt['report'],nfile)
    nlist=check_if_file_available(persons_dict,preport,overwrite=False)
    
    for tlist in nlist:

      try:
        download ( tlist,persons_dict, dest_folder=pt['dir_root'] )
        
        status='Available Locally'
      except requests.exceptions.SSLError:
          status='Not Available Locally'
    
      _progress(preport,tlist,status)
    



# First is to download all file from the web. Comment the line if all files has been downloaded
# _get_file_web()

