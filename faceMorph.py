#!/usr/bin/env python


import numpy as np
import cv2
import sys
import imageio
import dlib

# 从txt文件中读取坐标点
# path:路径名
def readPoints(path) :
    points = []
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))
    return points

# 通过srcTri和dstTri计算得到变换矩阵
# 对src进行仿射变换，输出的图片大小为size
#src:原图片 srcTri:原图片三个点的坐标 dstTri:目标图片三个点的坐标 size为输出的尺寸
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    #根据三个点得到变换矩阵
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    #将仿射变换用于src
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


#img1:图片1，矩阵  img2:图片2，矩阵  imgMorph:融合后图片，相当于一个画布，矩阵
#t1:图片1的一个三角形坐标  t2：图片2的一个三角形坐标  alpha:权重
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :
    # 查找包覆每个三角形的正矩形
    # 输入：2D点集。输出：(x,y,w,h)
    # 分别代表左上点坐标(x,y)，宽高(w,h)
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))  
    #计算三角形坐标点相对于矩形左上点的偏移，得到的是在矩形框内三角形的坐标
    t1Rect = []
    t2Rect = []
    tRect = []
    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
    # 在矩形的mask中画出三角形,三角形内部值为1，外部值为0

    mask = np.zeros((r[3], r[2], 3),dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    #取出图片1和图片2各自的矩形图片，矩阵
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    #融合图片的小矩形的尺寸
    size = (r[2], r[3])
    #对图片1和图片2的矩形做仿射变换
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)
    #对两个矩形进行融合
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2
    #1-mask使三角形内全为0，三角形外全为1
    #保留三角形外的像素值，添加三角形内的像素值
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask
    
#生成gif文件
#image_list:图片列表  gif_name:输出的gif文件名 duration:两张图的变换间隔，单位s 
def creat_gif(image_list, gif_name, duration = 2):
        frames = []
        for image_name in image_list:
            frames.append(imageio.imread(image_name))
        imageio.mimsave(gif_name, frames, 'GIF', duration = duration)

#生成人脸特征点
def generate_feature(imagename,outputname):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    img_rd = cv2.imread(imagename)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    faces = detector(img_gray, 0)
    if len(faces) != 0:
    # 检测到人脸
        for i in range(len(faces)):
        # 取特征点坐标
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[i]).parts()])
            f = open(outputname,'a')
            for idx, point in enumerate(landmarks):
            # 68 点的坐标
                pos = (point[0, 0], point[0, 1])
                msg = str(pos[0])+' '+str(pos[1])+'\n'
                f.write(msg)
            ad = str(0)+' '+str(0)+'\n'
            f.write(ad)
            ad = str(0)+' '+str(img_rd.shape[0]-1)+'\n'
            f.write(ad)
            ad = str(0)+' '+str(int(img_rd.shape[0]/2))+'\n'
            f.write(ad)
            ad = str(int(img_rd.shape[1]/2))+' '+str(img_rd.shape[0]-1)+'\n'
            f.write(ad)
            ad = str(img_rd.shape[1]-1)+' '+str(img_rd.shape[0]-1)+'\n'
            f.write(ad)

            ad = str(img_rd.shape[1]-1)+' '+str(img_rd.shape[0]-1)+'\n'
            f.write(ad)
            ad = str(img_rd.shape[1]-1)+' '+str(int(img_rd.shape[0]/2))+'\n'
            f.write(ad)
            ad = str(img_rd.shape[1]-1)+' '+str(0)+'\n'
            f.write(ad)
            ad = str(int(img_rd.shape[1]/2))+' '+str(0)+'\n'
            f.write(ad)

            f.close()

#生成Delaunay三角剖分文件与特征点文件的关联文件
def change(filename):
    filename1 = 'first'
    points =[]
    with open('./file/'+filename1 + '.txt') as file :
        for line in file :
            x_m1,x_m2,y_m1,y_m2,z_m1,z_m2 = line.split()
            points.append(((int(x_m1),int(x_m2)), (int(y_m1),int(y_m2)), (int(z_m1),int(z_m2))))
    
    filename2 = filename
    features = []
    with open('./file/'+filename2 + '.txt') as file :
        for line in file :
            x,y = line.split()
            features.append((int(x),int(y)))
    final =[]
    for point in points:
        l = []
        for pp in point:
            index = features.index(pp)
            l.append(index)
        final.append(l)
    f1 = open('./file/'+'final.txt','a')
    for f in final:
        msg = str(f[0])+' '+str(f[1])+' '+str(f[2])+'\n'
        f1.write(msg)
    f1.close()



if __name__ == '__main__' :

    filename1 = 'image1.jpg'
    filename2 = 'image2.jpg'
    alphas = [0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1]
    # 读取图像数据
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    
    # 生成特征点文件
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    outputname1 = './file/'+filename1+'.txt'
    outputname2 = './file/'+filename2+'.txt'
    generate_feature(filename1,outputname1)
    generate_feature(filename2,outputname2)
    # 读取图片的特征点
    points1 = readPoints('./file/'+filename1 + '.txt')
    points2 = readPoints('./file/'+filename2 + '.txt')
    image_list = []
    change_list =[]
    #生成图片1的Delaunay三角剖分文件，文件名为first.txt
    rect = (0, 0, img1.shape[1], img1.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    point_tri = []
    with open('./file/'+filename1 + '.txt') as file :
        for line in file :
            x, y = line.split()
            point_tri.append((int(x), int(y)))
    for p in point_tri :
        subdiv.insert(p)
    triangleList = subdiv.getTriangleList()
    for t in triangleList :       
        msg = str(int(t[0])) +' '+ str(int(t[1])) +' '+ str(int(t[2])) +' '+ str(int(t[3])) +' ' +str(int(t[4]))+' '+ str(int(t[5]))+'\n'
        f = open('./file/'+'first.txt','a')
        f.write(msg)
    f.close()
    #转换上一步得到的三角剖分文件,生成final.txt文件
    change('image1.jpg')

    #对每一个alpha都生成一幅图片
    for alpha in alphas:
        points = []
    # 计算加权平均点坐标
        for i in range(0, len(points1)):
            x = ( 1 -alpha ) * points1[i][0] + alpha * points2[i][0]
            y = ( 1 -alpha ) * points1[i][1] + alpha * points2[i][1]
            points.append((x,y))
        #读取final.txt
        #每行的数分别表示特征点文件的第几行
        #38 40 37分别表示第38行，40行，37行的特征点

        with open('./file/'+'final.txt') as file :
            imgMorph = np.zeros(img1.shape, dtype = img1.dtype)
            index=0
            for line in file :
                index +=1
                x,y,z = line.split()
                x = int(x)
                y = int(y)
                z = int(z)
                t1 = [points1[x], points1[y], points1[z]]
                t2 = [points2[x], points2[y], points2[z]]
                t = [ points[x], points[y], points[z] ]
                #每次融合一个三角形
                morphTriangle(img1, img2, imgMorph, t1, t2, t,alpha)
                #测试用，看出融合的过程
                if alpha == 0.5:
                    cv2.imwrite('./change/'+str(index)+'.png',np.uint8(imgMorph))
                    change_list.append('./change/'+str(index)+'.png')
            #生成融合后的图片
            name = str(alpha*10)+'.png'
            image_list.append('./image/'+name)
            cv2.imwrite('./image/'+name,np.uint8(imgMorph))
    #生成gif文件
    gif_name = './output/new.gif'
    duration = 0.1
    creat_gif(image_list, gif_name, duration)
    creat_gif(change_list, './output/change.gif', duration)
    print('finish')
