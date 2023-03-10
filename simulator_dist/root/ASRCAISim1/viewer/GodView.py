# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

from math import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import pygame
from pygame.locals import QUIT
import numpy as np
from ASRCAISim1.utility.GraphicUtility import *
from ASRCAISim1.libCore import *

class GodView(Viewer):
	"""pygameを用いて、画面の上側に戦域を上から見た図を、下側に戦域を南から見た図を描画する例。
	機体、誘導弾、センサの覆域、防衛ラインの描画に加え、現在の時刻、得点、報酬、Agentの情報等を表示する。
	仮実装であり、特に無理やり１画面に二つの図を描いているため、誘導弾が隣の図にはみ出したり、センサ覆域がはみ出したりする。
	(覆域はなるべくはみ出さないように切断して必要な部分だけ描画するよう試みてはいる)
	"""
	def __init__(self,modelConfig,instanceConfig):
		Viewer.__init__(self,modelConfig,instanceConfig)
		self.width=1280#1920
		self.height=720#1080
	def validate(self):
		self.fps=60#フレームレート。現在は無効にしている。
		self.fieldMargin=np.array([0.1,0.1,0.1])#各軸の表示範囲の余裕。
		self.xyScaleType='same'#x軸とy軸の縮尺。sameとしたら同じになる。fitとしたら個別に拡大・縮小される。
		self.regionType=['fix','fix','fix']#各軸の表示範囲。fixとしたら固定。fitとしたら全機が入るように拡大・縮小される。
		self.w_margin=0.01
		self.h_margin=0.01
		self.h_xy=0.64
		self.h_yz=0.33
		self.isValid=True
		pygame.init()
		self.font=pygame.font.Font('freesansbold.ttf',12)
		self.window=pygame.display.set_mode((self.width,self.height),pygame.DOUBLEBUF|pygame.OPENGL|pygame.OPENGLBLIT)
		Draw2D.setShape(self.width,self.height)
		self.clock=pygame.time.Clock()
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	def close(self):
		pygame.quit()
	def onEpisodeBegin(self):
		"""エピソードの開始時(reset関数の最後)に呼ばれる。
		"""
		if(not self.isValid):
			self.validate()
		self.display()
	def onStepBegin(self):
		"""gym.Envとしてのstepの開始時に呼ばれる。
		"""
		pass
	def onStepEnd(self):
		"""gym.Envとしてのstepの最後に呼ばれる。
		Managerは得点計算⇛報酬計算⇛その他コールバック⇛画面表示の順で呼ぶ。
		"""
	def onInnerStepBegin(self):
		"""インナーループの各ステップの開始時(controlの前)に呼ばれる。
		"""
		for e in pygame.event.get():
			if(e.type==QUIT):
				self.manager.manualDone=True
	def onInnerStepEnd(self):
		"""#インナーループの各ステップの最後(perceiveの後)に呼ばれる。
		"""
		if(self.isValid):
			self.display()
	def onEpisodeEnd(self):
		"""エピソードの終了時(step関数の最後でdone==Trueの場合)に呼ばれる
		"""
		if(self.manager.manualDone):
			pygame.quit()
	def calcRegion(self):
		"""x,y,z軸の描画対象範囲を計算する。
		"""
		ruler=self.manager.getRuler()()
		self.fgtrRegion=[
			np.min(np.r_[[f().posI() for f in self.manager.getAssets(lambda a:isinstance(a,Fighter))]],0),
			np.max(np.r_[[f().posI() for f in self.manager.getAssets(lambda a:isinstance(a,Fighter))]],0)
		]
		ruler=self.manager.getRuler()()
		self.fieldRegion=[
			np.array([-ruler.dOut,-ruler.dLine,-ruler.hLim]),
			np.array([ruler.dOut,ruler.dLine,0])
		]
		xyAspect=self.h_xy*Draw2D.height/(Draw2D.width*(1-self.w_margin*2))
		minR=np.array([(self.fgtrRegion[0][i] if self.regionType[i]=='fit' else self.fieldRegion[0][i]) for i in range(3)])
		maxR=np.array([(self.fgtrRegion[1][i] if self.regionType[i]=='fit' else self.fieldRegion[1][i]) for i in range(3)])
		mid=(minR+maxR)/2.0
		delta=(maxR-minR)/2.0
		minR=mid-delta*(self.fieldMargin+np.array([1,1,1]))
		maxR=mid+delta*(self.fieldMargin+np.array([1,1,1]))
		if(self.xyScaleType=='same'):
			if(xyAspect*(maxR[1]-minR[1])>(maxR[0]-minR[0])):
				mid=(maxR[0]+minR[0])/2.0
				delta=xyAspect*(maxR[1]-minR[1])/2.0
				minR[0]=mid-delta
				maxR[0]=mid+delta
			elif(xyAspect*(maxR[1]-minR[1])<(maxR[0]-minR[0])):
				mid=(maxR[1]+minR[1])/2.0
				delta=(maxR[0]-minR[0])/xyAspect/2.0
				minR[1]=mid-delta
				maxR[1]=mid+delta
		self.region=[minR,maxR]
	def simToReg(self,sim):#returns (xy,yz)
		"""シミュレーション中の慣性座標系で表された位置simを、
		x,y,z各軸を描画範囲で正規化し0〜1にして返す。
		"""
		return (sim-self.region[0])/(self.region[1]-self.region[0])
	def regToSur(self,reg):#'xy' or 'yz'
		"""各軸正規化された位置regを、
		xy図、yz図上で対応する位置に表示されるような、画面全体のxy座標に変換して返す。
		"""
		xy=[
			Draw2D.width*(self.w_margin+reg[1]*(1.0-2*self.w_margin)),
			Draw2D.height*((self.h_margin*2+self.h_yz)+reg[0]*(self.h_xy))
		]
		yz=[
			Draw2D.width*(self.w_margin+reg[1]*(1.0-2*self.w_margin)),
			Draw2D.height*(self.h_margin+(1.0-reg[2])*self.h_yz)
		]
		return (xy,yz)
	def makeGrid(self,interval):
		"""戦域の区切り線を描く。
		"""
		lower=np.ceil(self.region[0]/interval)
		upper=np.floor(self.region[1]/interval)
		cnt=upper-lower
		for i in range(int(lower[0]),int(upper[0])+1):
			d=(i*interval[0]-self.region[0][0])/(self.region[1][0]-self.region[0][0])
			p=[self.regToSur(x)[0] for x in [[d,0,0],[d,1,1]]]
			drawLine2D(p[0][0],p[0][1],p[1][0],p[1][1])
		for i in range(int(lower[1]),int(upper[1])+1):
			d=(i*interval[1]-self.region[0][1])/(self.region[1][1]-self.region[0][1])
			p=[self.regToSur(x) for x in [[0,d,0],[1,d,1]]]
			drawLine2D(p[0][0][0],p[0][0][1],p[1][0][0],p[1][0][1])
			drawLine2D(p[0][1][0],p[0][1][1],p[1][1][0],p[1][1][1])
		for i in range(int(lower[2]),int(upper[2])+1):
			d=(i*interval[2]-self.region[0][2])/(self.region[1][2]-self.region[0][2])
			p=[self.regToSur(x)[1] for x in [[0,0,d],[1,1,d]]]
			drawLine2D(p[0][0],p[0][1],p[1][0],p[1][1])
	def display(self):
		"""画面描画を行う。
		"""
		self.calcRegion()
		glClearColor(0.8, 0.8, 0.8, 1.0)
		glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT)
		glMatrixMode(GL_MODELVIEW)
		Draw2D.begin()
		ruler=self.manager.getRuler()()
		#描画開始
		glColor4f(1.0,1.0,1.0,1)
		p1,p2=self.regToSur([0,0,0]),self.regToSur([1,1,1])
		fillRect2D(p1[0][0],p1[0][1],p2[0][0]-p1[0][0],p2[0][1]-p1[0][1])
		fillRect2D(p1[1][0],p1[1][1],p2[1][0]-p1[1][0],p2[1][1]-p1[1][1])
		glColor4f(0.6,0.6,0.6,1)
		self.makeGrid([20000,20000,4000])
		glLineWidth(3.0)
		#防衛ライン、南北境界線
		nw=self.regToSur(self.simToReg(np.array([+ruler.dOut,-ruler.dLine,0])))
		ne=self.regToSur(self.simToReg(np.array([+ruler.dOut,+ruler.dLine,0])))
		sw=self.regToSur(self.simToReg(np.array([-ruler.dOut,-ruler.dLine,0])))
		se=self.regToSur(self.simToReg(np.array([-ruler.dOut,+ruler.dLine,0])))
		uw=self.regToSur(self.simToReg(np.array([+ruler.dOut,-ruler.dLine,-ruler.hLim])))
		ue=self.regToSur(self.simToReg(np.array([+ruler.dOut,+ruler.dLine,-ruler.hLim])))
		glColor4f(0,0,1,1)
		d=(ruler.dLine-self.region[0][1])/(self.region[1][1]-self.region[0][1])
		p=[self.regToSur(x) for x in [[0,d,0],[1,d,1]]]
		drawLine2D(ne[0][0],ne[0][1],se[0][0],se[0][1])
		drawLine2D(ne[1][0],ne[1][1],ue[1][0],ue[1][1])
		glColor4f(1,0,0,1)
		drawLine2D(nw[0][0],nw[0][1],sw[0][0],sw[0][1])
		drawLine2D(nw[1][0],nw[1][1],uw[1][0],uw[1][1])
		glColor4f(0,0,0,1)
		drawLine2D(nw[0][0],nw[0][1],ne[0][0],ne[0][1])
		drawLine2D(nw[1][0],nw[1][1],ne[1][0],ne[1][1])
		drawLine2D(sw[0][0],sw[0][1],se[0][0],se[0][1])
		drawLine2D(uw[1][0],uw[1][1],ue[1][0],ue[1][1])
		glLineWidth(1.0)
		txt="Time:{:6.1f}s".format(self.manager.getTime())
		pos=(30,self.height-24-10)
		drawText2D(txt,pygame.font.Font('freesansbold.ttf',16),pos,(0,180,50,255))
		txt="Score:"+",".join([team+"{:.2f}".format(score) for team,score in self.manager.scores.items()])
		pos=(30,self.height-24-40)
		drawText2D(txt,pygame.font.Font('freesansbold.ttf',16),pos,(0,180,50,255))
		west=[agent().__repr__() for agent in self.manager.getAgents(lambda a:a.getTeam()==ruler.westSider)]
		east=[agent().__repr__() for agent in self.manager.getAgents(lambda a:a.getTeam()==ruler.eastSider)]
		#txt="West( "+self.manager.ruler.westSider+" ):"+str(west)
		#pos=(30,self.height-24-70)
		#drawText2D(txt,pygame.font.Font('freesansbold.ttf',14),pos,(255,150,150,255))
		#txt="East( "+self.manager.ruler.eastSider+" ):"+str(east)
		#pos=(30,self.height-24-90)
		#drawText2D(txt,pygame.font.Font('freesansbold.ttf',14),pos,(150,150,255,255))
		#報酬
		west=["{:.1f}".format(self.manager.totalRewards[agent().getFullName()]) for agent in self.manager.getAgents(lambda a:a.getTeam()==ruler.westSider)]
		east=["{:.1f}".format(self.manager.totalRewards[agent().getFullName()]) for agent in self.manager.getAgents(lambda a:a.getTeam()==ruler.eastSider)]
		txt="Total reward of West( "+ruler.westSider+" ):"+str(west)
		pos=(30,self.height-24-130)
		drawText2D(txt,pygame.font.Font('freesansbold.ttf',14),pos,(255,50,50,255))
		txt="Total reward of East( "+ruler.eastSider+" ):"+str(east)
		pos=(30,self.height-24-150)
		drawText2D(txt,pygame.font.Font('freesansbold.ttf',14),pos,(50,50,255,255))
		#機体
		leading={ruler.westSider:-ruler.dLine,ruler.eastSider:-ruler.dLine}
		for f in self.manager.getAssets(lambda a:a.getTeam()==ruler.eastSider and isinstance(a,Fighter)):
			f=f()
			if(f.isAlive()):
				leading[ruler.eastSider]=max(np.dot(ruler.forwardAx[ruler.eastSider],f.posI()[0:2]),leading[ruler.eastSider])
				pf=self.regToSur(self.simToReg(f.posI()))
				glColor4f(0,0,1,1)
				drawCircle2D(pf[0],5,8)
				drawCircle2D(pf[1],5,8)
				ag=f.agent()
				agTxt=""
				agOb=ag.observables()[f.getFullName()]
				if("state" in agOb):
					agTxt=",state="+agOb["state"]
				V=int(round(np.linalg.norm(f.velI())))
				drawText2D(f.getName()+":v="+str(V)+",m="+str(f.remMsls)+agTxt,self.font,(pf[0][0]+10,pf[0][1]+0),(0,0,0,255))
				drawText2D(f.getName()+":v="+str(V)+",m="+str(f.remMsls)+agTxt,self.font,(pf[1][0]+10,pf[1][1]+0),(0,0,0,255))
				#センサ
				glColor4f(0.4,0.4,1.0,0.2)
				ex=f.relBtoI(np.array([1.,0.,0.]))
				ey=f.relBtoI(np.array([0.,1.,0.]))
				ez=f.relBtoI(np.array([0.,0.,1.]))
				xy=sqrt(ex[0]*ex[0]+ex[1]*ex[1])
				pitchAngle=atan2(-ex[2],xy)
				eyHorizontal=np.cross(np.array([0.,0.,1.]),ex)
				sn=np.linalg.norm(eyHorizontal)
				if(abs(sn)<1e-6):
					eyHorizontal=np.array([0.,1.,0.])
				else:
					eyHorizontal/=sn
				exHorizontal=np.cross(eyHorizontal,np.array([0.,0.,1.]))
				exHorizontal/=np.linalg.norm(exHorizontal)
				cs=np.dot(ex,exHorizontal)
				L=f.radar().Lref
				N=16
				#XY平面プラス側
				psxy=[self.simToReg(f.posI())]+[self.simToReg(
					f.posI()+L*(cos(-pi/2+pi*i/N)*exHorizontal+sin(-pi/2+pi*i/N)*eyHorizontal))
					for i in range(N+1)]
				psxy=polygonRegionCut(psxy,[0.0,1.0],0)
				glBegin(GL_TRIANGLE_FAN)
				for p in psxy:
					tmp = Draw2D.convertPos(self.regToSur(p)[0])
					glVertex2d(tmp[0], tmp[1])
				glEnd()
				#XY平面マイナス側
				psxy=[self.simToReg(f.posI())]+[self.simToReg(
					f.posI()-L*(cos(-pi/2+pi*i/N)*exHorizontal*abs(sin(pitchAngle))+sin(-pi/2+pi*i/N)*eyHorizontal))
					for i in range(N+1)]
				psxy=polygonRegionCut(psxy,[0.0,1.0],0)
				glBegin(GL_TRIANGLE_FAN)
				for p in psxy:
					tmp = Draw2D.convertPos(self.regToSur(p)[0])
					glVertex2d(tmp[0], tmp[1])
				glEnd()
				exInYZ=(ex*[0,1,1])
				sn=np.linalg.norm(exInYZ)
				if(sn<1e-6):
					exInYZ=np.array([0.,1.,0.])
					absSinYaw=1.0
					eyz=np.array([0.,0.,1.])
				else:
					exInYZ=exInYZ/sn
					eyz=np.cross(exInYZ,np.array([1.,0.,0.]))
					absSinYaw=abs(asin(ex[0]))
				#YZ平面プラス側
				psyz=[self.simToReg(f.posI())]+[self.simToReg(
					f.posI()+L*(cos(-pi/2+pi*i/N)*exInYZ+sin(-pi/2+pi*i/N)*eyz))
					for i in range(N+1)]
				psyz=polygonRegionCut(psyz,[0.0,1.0],2)
				glBegin(GL_TRIANGLE_FAN)
				for p in psyz:
					tmp = Draw2D.convertPos(self.regToSur(p)[1])
					glVertex2d(tmp[0], tmp[1])
				glEnd()
				#YZ平面マイナス側
				psyz=[self.simToReg(f.posI())]+[self.simToReg(
					f.posI()-L*(cos(-pi/2+pi*i/N)*exInYZ*absSinYaw+sin(-pi/2+pi*i/N)*eyz))
					for i in range(N+1)]
				psyz=polygonRegionCut(psyz,[0.0,1.0],2)
				glBegin(GL_TRIANGLE_FAN)
				for p in psyz:
					tmp = Draw2D.convertPos(self.regToSur(p)[1])
					glVertex2d(tmp[0], tmp[1])
				glEnd()
		glColor4f(1,0,0,1)
		for f in self.manager.getAssets(lambda a:a.getTeam()==ruler.westSider and isinstance(a,Fighter)):
			f=f()
			if(f.isAlive()):
				leading[ruler.westSider]=max(np.dot(ruler.forwardAx[ruler.westSider],f.posI()[0:2]),leading[ruler.westSider])
				pf=self.regToSur(self.simToReg(f.posI()))
				glColor4f(1,0,0,1)
				drawCircle2D(pf[0],5,8)
				drawCircle2D(pf[1],5,8)
				ag=f.agent()
				agTxt=""
				agOb=ag.observables()[f.getFullName()]
				if("state" in agOb):
					agTxt=",state="+agOb["state"]
				V=int(round(np.linalg.norm(f.velI())))
				drawText2D(f.getName()+":v="+str(V)+",m="+str(f.remMsls)+agTxt,self.font,(pf[0][0]+10,pf[0][1]+0),(0,0,0,255))
				drawText2D(f.getName()+":v="+str(V)+",m="+str(f.remMsls)+agTxt,self.font,(pf[1][0]+10,pf[1][1]+0),(0,0,0,255))
				#センサ
				glColor4f(1.0,0.4,0.4,0.2)
				ex=f.relBtoI(np.array([1.,0.,0.]))
				ey=f.relBtoI(np.array([0.,1.,0.]))
				ez=f.relBtoI(np.array([0.,0.,1.]))
				xy=sqrt(ex[0]*ex[0]+ex[1]*ex[1])
				pitchAngle=atan2(-ex[2],xy)
				eyHorizontal=np.cross(np.array([0.,0.,1.]),ex)
				sn=np.linalg.norm(eyHorizontal)
				if(abs(sn)<1e-6):
					eyHorizontal=np.array([0.,1.,0.])
				else:
					eyHorizontal/=sn
				exHorizontal=np.cross(eyHorizontal,np.array([0.,0.,1.]))
				exHorizontal/=np.linalg.norm(exHorizontal)
				cs=np.dot(ex,exHorizontal)
				L=f.radar().Lref
				N=16
				#XY平面プラス側
				psxy=[self.simToReg(f.posI())]+[self.simToReg(
					f.posI()+L*(cos(-pi/2+pi*i/N)*exHorizontal+sin(-pi/2+pi*i/N)*eyHorizontal))
					for i in range(N+1)]
				psxy=polygonRegionCut(psxy,[0.0,1.0],0)
				glBegin(GL_TRIANGLE_FAN)
				for p in psxy:
					tmp = Draw2D.convertPos(self.regToSur(p)[0])
					glVertex2d(tmp[0], tmp[1])
				glEnd()
				#XY平面マイナス側
				psxy=[self.simToReg(f.posI())]+[self.simToReg(
					f.posI()-L*(cos(-pi/2+pi*i/N)*exHorizontal*abs(sin(pitchAngle))+sin(-pi/2+pi*i/N)*eyHorizontal))
					for i in range(N+1)]
				psxy=polygonRegionCut(psxy,[0.0,1.0],0)
				glBegin(GL_TRIANGLE_FAN)
				for p in psxy:
					tmp = Draw2D.convertPos(self.regToSur(p)[0])
					glVertex2d(tmp[0], tmp[1])
				glEnd()
				exInYZ=(ex*[0,1,1])
				sn=np.linalg.norm(exInYZ)
				if(sn<1e-6):
					exInYZ=np.array([0.,1.,0.])
					absSinYaw=1.0
					eyz=np.array([0.,0.,1.])
				else:
					exInYZ=exInYZ/sn
					eyz=np.cross(exInYZ,np.array([1.,0.,0.]))
					absSinYaw=abs(asin(ex[0]))
				#YZ平面プラス側
				psyz=[self.simToReg(f.posI())]+[self.simToReg(
					f.posI()+L*(cos(-pi/2+pi*i/N)*exInYZ+sin(-pi/2+pi*i/N)*eyz))
					for i in range(N+1)]
				psyz=polygonRegionCut(psyz,[0.0,1.0],2)
				glBegin(GL_TRIANGLE_FAN)
				for p in psyz:
					tmp = Draw2D.convertPos(self.regToSur(p)[1])
					glVertex2d(tmp[0], tmp[1])
				glEnd()
				#YZ平面マイナス側
				psyz=[self.simToReg(f.posI())]+[self.simToReg(
					f.posI()-L*(cos(-pi/2+pi*i/N)*exInYZ*absSinYaw+sin(-pi/2+pi*i/N)*eyz))
					for i in range(N+1)]
				psyz=polygonRegionCut(psyz,[0.0,1.0],2)
				glBegin(GL_TRIANGLE_FAN)
				for p in psyz:
					tmp = Draw2D.convertPos(self.regToSur(p)[1])
					glVertex2d(tmp[0], tmp[1])
				glEnd()
		#誘導弾
		for m in self.manager.getAssets(lambda a:a.getTeam()==ruler.eastSider and isinstance(a,Missile)):
			m=m()
			if(m.isAlive() and m.hasLaunched):
				if(m.mode==Missile.Mode.SELF):
					glColor4f(1.0,1.0,0.0,1.0)
				elif(m.mode==Missile.Mode.GUIDED):
					if(m.sensor().isActive):
						glColor4f(0.0,0.8,1,1)
					else:
						glColor4f(0,0,1,1)
				else:
					if(m.sensor().isActive):
						glColor4f(0.0,0.8,1,1)
					else:
						glColor4f(0.7,0.7,1,1)
				pm=self.regToSur(self.simToReg(m.posI()))
				fillCircle2D(pm[0],5,8)
				fillCircle2D(pm[1],5,8)
				#目標
				tm=self.regToSur(self.simToReg(m.estTPos))
				fillCircle2D(tm[0],3,8)
				fillCircle2D(tm[1],3,8)
				drawLine2D(pm[0][0],pm[0][1],tm[0][0],tm[0][1])
				drawLine2D(pm[1][0],pm[1][1],tm[1][0],tm[1][1])
				#センサ
				if(m.sensor().isActive):
					glColor4f(1.0,1.0,0.0,0.2)
					ex=m.relBtoI(np.array([1.,0.,0.]))
					ey=m.relBtoI(np.array([0.,1.,0.]))
					ez=m.relBtoI(np.array([0.,0.,1.]))
					L=m.sensor().Lref
					angle=m.sensor().thetaFOR
					N=16
					psxy=[self.regToSur(self.simToReg(
						m.posI()+L*(cos(-angle+angle*2*i/N)*ex+sin(-angle+angle*2*i/N)*ey)))[0]
						for i in range(N+1)]
					psyz=[self.regToSur(self.simToReg(
						m.posI()+L*(cos(-angle+angle*2*i/N)*ez+sin(-angle+angle*2*i/N)*ey)))[1]
						for i in range(N+1)]
					glBegin(GL_TRIANGLE_FAN)
					tmp = Draw2D.convertPos(pm[0])
					glVertex2d(tmp[0],tmp[1])
					for i in range(N+1):
						tmp = Draw2D.convertPos(psxy[i])
						glVertex2d(tmp[0], tmp[1])
					glEnd()
		glColor4f(1,0,0,1)
		for m in self.manager.getAssets(lambda a:a.getTeam()==ruler.westSider and isinstance(a,Missile)):
			m=m()
			if(m.isAlive() and m.hasLaunched):
				if(m.mode==Missile.Mode.SELF):
					glColor4f(1,1,0,1)
				elif(m.mode==Missile.Mode.GUIDED):
					if(m.sensor().isActive):
						glColor4f(1.0,0.,0.8,1)
					else:
						glColor4f(1,0,0,1)
				else:
					if(m.sensor().isActive):
						glColor4f(1.0,0.,0.8,1)
					else:
						glColor4f(1,0.7,0.7,1)
				pm=self.regToSur(self.simToReg(m.posI()))
				fillCircle2D(pm[0],5,8)
				fillCircle2D(pm[1],5,8)
				#目標
				tm=self.regToSur(self.simToReg(m.estTPos))
				fillCircle2D(tm[0],3,8)
				fillCircle2D(tm[1],3,8)
				drawLine2D(pm[0][0],pm[0][1],tm[0][0],tm[0][1])
				drawLine2D(pm[1][0],pm[1][1],tm[1][0],tm[1][1])
				#センサ
				if(m.sensor().isActive):
					glColor4f(1.0,1.0,0.0,0.2)
					ex=m.relBtoI(np.array([1.,0.,0.]))
					ey=m.relBtoI(np.array([0.,1.,0.]))
					ez=m.relBtoI(np.array([0.,0.,1.]))
					L=m.sensor().Lref
					angle=m.sensor().thetaFOR
					N=16
					psxy=[self.regToSur(self.simToReg(
						m.posI()+L*(cos(-angle+angle*2*i/N)*ex+sin(-angle+angle*2*i/N)*ey)))[0]
						for i in range(N+1)]
					psyz=[self.regToSur(self.simToReg(
						m.posI()+L*(cos(-angle+angle*2*i/N)*ez+sin(-angle+angle*2*i/N)*ey)))[1]
						for i in range(N+1)]
					glBegin(GL_TRIANGLE_FAN)
					tmp = Draw2D.convertPos(pm[0])
					glVertex2d(tmp[0],tmp[1])
					for i in range(N+1):
						tmp = Draw2D.convertPos(psxy[i])
						glVertex2d(tmp[0], tmp[1])
					glEnd()
		#前線の中心
		glColor4f(0,0.6,0,1)
		glLineWidth(3.0)
		center=(leading[ruler.westSider]-leading[ruler.eastSider])/2
		n=self.regToSur(self.simToReg(np.array([+ruler.dOut,center,0])))
		s=self.regToSur(self.simToReg(np.array([-ruler.dOut,center,0])))
		drawLine2D(n[0][0],n[0][1],s[0][0],s[0][1])
		txt=str(round(center/1000.0))
		drawText2D(txt,pygame.font.Font('freesansbold.ttf',16),(n[0][0],n[0][1]+5),(0,int(255*0.6),0,255))
		glLineWidth(1.0)
		#描画終了
		Draw2D.end()
		glFlush()
		pygame.display.flip()
		#pygame.display.update()
		#self.clock.tick(self.fps)
