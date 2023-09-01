import cv2 as cv2
import numpy as np


mark = np.array([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,  0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,  0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255],
 [255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255],
 [255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255],
 [255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255],
 [255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255],
 [255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255],
 [255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255],
 [255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255],
 [255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,  255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,  255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,  255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,  255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,  0,   0,   0,   0,   0,   255,  255, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,  0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]],dtype=np.uint8)

struc = np.array([[255,255,255,255,255],
                 [255,255,255,255,255],
                 [255,255,255,255,255],
                 [255,255,255,255,255],
                 [255,255,255,255,255]],dtype=np.uint8)
def piecewiseLinTransform(k1,k2,a,b,image):
    img = image.copy()
   
    for j in range(np.size(k1)):
        where = np.where((img >= a[j]) & (img < b[j]))
        img[where] = np.multiply(img[where],k1[j])
        img[where] = np.add(img[where],k2[j])
    
    return(img)


def dilation(image,mask):
    img = image.copy()
    mask_size = np.shape(mask)
    a = mask
    # where_1 = np.where(a == 255)
    t = int(mask_size[0]/2)
    size = np.shape(img)
    output = np.zeros(size,dtype=np.uint8)

    for i in range(t,size[0]-t ):
        for j in range(t,size[1]-t):
            # b = img[i-t:i+t+1,j-t:j+t+1]
            c = output[i-t:i+t+1,j-t:j+t+1]
            if(img[i,j]==255):
                output[i-t:i+t+1,j-t:j+t+1] = a

    
    return output

def erosion(image,mask):
    img = image.copy()
    mask_size = np.shape(mask)
    a = mask
    # where_1 = np.where(a == 255)
    t = int(mask_size[0]/2)
    size = np.shape(img)
    output = np.zeros(size,dtype=np.uint8)

    for i in range(t,size[0]-t ):
        for j in range(t,size[1]-t):
            b = img[i-t:i+t+1,j-t:j+t+1]
            # x = np.unique(b[where_1])
            if(np.array_equal(a,b)):
                output[i,j] = 255
    
    return output

def hit_miss(spot,kernel):
    if(np.shape(spot) != np.shape(kernel)): return False
    where_0 = np.where(kernel == 0)   
    uniq = np.unique(spot[where_0])
    if(len(uniq) == 2): return False
    if(uniq[0] == 255): return False
    if(uniq[0] == 0): return True
    
    return False

def getAnswers(omr_sheet)->list:
  """
     Your documentation here
  """
  omr_sheet = cv2.cvtColor(omr_sheet,cv2.COLOR_RGB2GRAY)
  k1 = [0,0]
  k2 = [0,255]
  a = [0,156]
  b = [157,255]
  anskey_thres = piecewiseLinTransform(k1,k2,a,b,omr_sheet)
  out1 = dilation(anskey_thres,struc)
  anskey_thres = erosion(out1,struc)
  anskey_thres = erosion(anskey_thres,struc)
  
  start_v = 799
  h_cord = [222,559,896]
  sq_s = 27
  gap = 15
  gap_x = 15.4
  answers = [-1]*45
  l = 0
  for i in range(3):
    g = start_v
    x = start_v
    y = h_cord[i]
    for j in range(15):
      a = y 
      b = a + sq_s + gap
      c = b + sq_s + gap
      d = c + sq_s + gap
      g = (g + gap_x + sq_s)
      if(hit_miss(anskey_thres[x:x+sq_s,a:a+sq_s],mark)): answers[l] = 'A'
      if(hit_miss(anskey_thres[x:x+sq_s,b:b+sq_s],mark)): answers[l] = 'B'
      if(hit_miss(anskey_thres[x:x+sq_s,c:c+sq_s],mark)): answers[l] = 'C'
      if(hit_miss(anskey_thres[x:x+sq_s,d:d+sq_s],mark)): answers[l] = 'D'
      x = int(g)
      l = l+1
  # do all your processing here.
  # return answers of particular omr sheet here
  
  return answers

if __name__ == "__main__":
  

  T = int(input().strip())
                           
  
  for i in range(T):
    
    fileName = input().strip() 
    omr_sheet = cv2.imread(fileName)
    
    
    answers = getAnswers(omr_sheet)
    for answer in answers: 
      print(answer) 