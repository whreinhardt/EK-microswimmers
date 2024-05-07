import numpy
import pylab
from scipy.integrate import odeint


def arccoth(x):
	return 0.5*numpy.log((x+1.0)/(x-1.0))

def vel_full(params,t, a, b, phi, w):
	hm=params[0]
	alpha = params[1]
	h0=hm+alpha/2.0
	hp = h0+alpha/2.0
	phi = phi/(1+alpha**2)**0.5
	I1= numpy.log(hp/hm)*(1/alpha)
	I2=4.0/(4*h0**2-alpha**2)
	I1x=1/alpha -2.0*h0*arccoth(2*h0/alpha)/alpha**2
	I2x=(2.0/alpha**2)*(arccoth(2*h0/alpha)-2*h0*alpha/(4*h0**2-alpha**2))
	G, sn, sx, sxx=getP(h0, alpha, w, N=30)
	Ginv = numpy.linalg.inv(G)
	I3=numpy.dot(sn, numpy.dot(Ginv, sn))
	I3x=numpy.dot(sx, numpy.dot(Ginv, sn))*0.5 + numpy.dot(sn, numpy.dot(Ginv, sx))*0.5
	I3xx=numpy.dot(sxx, numpy.dot(Ginv, sx))
	v=(-w*phi*I2*alpha/numpy.log(hp/hm)-alpha/2/(1-alpha**2)+b*phi)/(w*I1)
	lift = (w**3/6.)*I3*v*alpha/2.0-alpha*b*phi-1/(1-alpha**2)
	torque= (w**3/6.)*I3x*v*alpha/2.0+2*alpha*w*(I1x*v+I2x*phi*alpha/numpy.log(hp/hm))+a*phi
	M=(w**3/6.)*numpy.array([[I3,I3x],[I3x, I3xx]])
	res=numpy.linalg.solve(M, numpy.array([lift, torque]))
	if numpy.any(numpy.isnan(res)):
		return [0,0]
	else:
		return [res[0]-res[1]*(1+alpha)/2.0, res[1]*(1+alpha)]

def finalPos(a, b, phi, w):
	t = [0, 30]
	soln=odeint(vel_full, [0.1, 0.1], t, args=(a, b, phi, w))
	return soln[-1]

def getP(h0, alpha, w, N=10):
	# we treat lubrication problems by expanding the presusre in sin(n*pi*x) functions
	# solving the reynold equation reduces to a matrix inverse where G is the matrix of sin functions integrated with an h^3 weight
	# sn are the integrals of all of the basis functions (i.e. the integrals of each sin term, enabling calculation of total pressure forces while sx and sxx are integrals of each derivative (which are useful in comnputing torque terms associated to translations and rotaiton)
	G = numpy.zeros((N,N))
	for m in range(1, N+1):
		for n in range(m, N+1):
			g_mn=n*m*numpy.pi**2*CC(m,n, alpha, h0)/6.0-SS(m,n, alpha, h0)/3.0
			G[m-1, n-1]=g_mn
			G[n-1, m-1]=g_mn
	sn = numpy.array([S(n) for n in range(1, N+1)])
	sx = numpy.array([Sx(n) for n in range(1, N+1)])
	sxx = numpy.array([Sxx(n) for n in range(1, N+1)])
	return G, sn, sx, sxx

def SS(m,n,alpha, h0):
	#integral of sin(n*pi*x)*sin(m*pi*x)
	if m==n:
		return (1/8.)*h0*(4*h0**2+(1-6.0/(n**2 *numpy.pi**2)) *alpha**2)
	res=-1/(2.0*(m - n)**4*(m + n)**4*numpy.pi**4)
	res=res*3*m*n*alpha*(-4*(-1 + (-1)**(m + n))*h0**2*(m**2 - n**2)**2*numpy.pi**2-4*(1+(-1)**(m+n))*h0*(m**2-n**2)**2*numpy.pi**2*alpha-(-1+(-1)**(m+n))*(-16*(m**2+n**2)+(m**2-n**2)**2 *numpy.pi**2)*alpha**2)
	return res

def CC(m,n, alpha, h0):
	if m==n:
		return (1/8.)*h0*(4*h0**2+(1+6/(n**2 *numpy.pi**2)) *alpha**2)
	res =1/(4.0*(m-n)**4*(m+n)**4*numpy.pi**4)
	res=res*3 *alpha*(4*(-1+(-1)**(m+n))*h0**2*(m**2-n**2)**2*(m**2+n**2)*numpy.pi**2+4*(1+(-1)**(m+n))*h0*(m**2 -n**2)**2*(m**2+n**2)*numpy.pi**2*alpha+(-1+(-1)**(m+n))*(-8*(m**4+6*m**2*n**2+n**4)+(m**2-n**2)**2*(m**2+n**2) *numpy.pi**2)*alpha**2)
	return res

def S(n):
	#integral of Sin(n*x*Pi) from x=0 to x=1
	res=-1*((-1+(-1)**n)/(n*numpy.pi))
	return res

def Sx(n):
	#integral of Dx(Sin(n*x*Pi)) from x=0 to x=1
	res= -1*((1+(-1)**n)/(2*n*numpy.pi))
	return res

def Sxx(n):
	#integral of Dxx(Sin(n*x*Pi)) from x=0 to x=1
	res=-(((-1+(-1)**n)*(-8 + n**2 *numpy.pi**2))/(4*n**3 *numpy.pi**3))
	return res

def buildP(h0, alpha, w, N=100):
	#Helper function to build the approximated pressure function under the robot
	x = numpy.linspace(-1/2., 1/2., 1000)
	basis = [numpy.sin(numpy.pi*n*(x+0.5)) for n in range(1, N+1)]
	G, sn, sx,sxx = getP(h0, alpha, w, N)
	an = numpy.linalg.solve(G, sn)
	p_i=x*0
	for i in range(0, N):
		p_i=p_i+an[i]*basis[i]
	return p_i
	
def error_func(p):
	scale=p[0]
	a=p[1]
	w=p[3]
	b=p[2]
	#hardcoded raw data from the publication
	vh=numpy.array([444.1,400.03,157.06,237.3,54.25,149,191.8,189.45,345.26,421.97,342.83])*scale
	ah=numpy.array([31.063,24.101,8.466,22.05,11.24,5.02,14.265,11.353,22.73,25.289,22.751])
	ah=numpy.tan(ah*3.14/180)
	vg=numpy.array([189.45,345.26,421.97,342.83,459.51,307.68,332.5])*scale
	gg=numpy.array([3.514,5.4645,6.075,7.2675,9.136,5.3275,5.1635])/400.0
	loss=0
	for k in range(0, len(vh)):
		pos=finalPos(a,b,-vh[k], w)
		loss = loss+(pos[1]-ah[k])**2
	for k in range(0, len(vg)):
		pos=finalPos(a,b,-vg[k], w)
		hm = pos[0]
		loss = loss+10*(hm-gg[k])**2
	return loss

def error_func_small(p, w):
	scale=p[0]
	a=p[1]
	b=0
	#hardcoded raw data from the publication
	vh=numpy.array([444.1,400.03,157.06,237.3,54.25,149,191.8,189.45,345.26,421.97,342.83])*scale
	ah=numpy.array([31.063,24.101,8.466,22.05,11.24,5.02,14.265,11.353,22.73,25.289,22.751])
	ah=numpy.tan(ah*3.14/180)
	vg=numpy.array([189.45,345.26,421.97,342.83,459.51,307.68,332.5])*scale
	gg=numpy.array([3.514,5.4645,6.075,7.2675,9.136,5.3275,5.1635])/400.0
	loss=0
	for k in range(0, len(vh)):
		pos=finalPos(a,b,-vh[k], w)
		loss = loss+(pos[1]-ah[k])**2
	for k in range(0, len(vg)):
		pos=finalPos(a,b,-vg[k], w)
		hm = pos[0]
		loss = loss+10*(hm-gg[k])**2
	return loss

if __name__ == "__main__":
	#example code showing how	

	# example of using the code to compute the evolution of the robot's body variables
	hm0, alpha0 =0.1, 0.1
	t = numpy.linspace(0, 10, 100)
	soln=odeint(vel_full, [hm0, alpha0], t, args=(0., 0., -1., 1/6.63),h0=1e-6)
	h, alpha = zip(*soln)
	pylab.plot(t, h)
	pylab.show()

	#example of using the code to scan final robot configurations after equillibrating when parameters are varried
	phi = numpy.linspace(0, 1, 20)
	phi = phi+phi[1]
	a,b,w=0.2, 0, 1/6.63
	model = numpy.array([finalPos(a, b, -p, w) for p in phi])
	#plot the gap height vs applied field for the specified parameters
	pylab.plot(phi, model[:, 0])
	pylab.show()