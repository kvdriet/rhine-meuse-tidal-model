import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy import integrate

class Network_model_RM():
    def __init__(self, Av = 0.005, Sf = 0.005, eta_riverflow = 0, u_riverflow = 0):
        self.T_M2 = 44700	 #Period M2 constituent [s] 44700
        self.omega = 2 * np.pi/self.T_M2 #Radian frequency M2 consitituent [1/s]
        self.Q = 10
        self.ZM2 = 0.80
        self.ZM4 = 0.2
        self.phi = 90/180*np.pi
        self.L_r = np.array((95e3,44.5e3))
        self.L_o = np.array((16.4,21.7)) * 1e3
        self.L_m = np.array((47.5,23.4,35.8)) * 1e3
        self.H_r = np.array((5,8))
        self.H_o = np.array((15,7.6))
        self.H_m = np.array((8,5,10))
        self.B_river = np.array((450,2200))
        self.B_ocean = np.array((700,331))     
        self.B_middle = np.array((200,420,320))
        self.B_m0 = self.B_middle
        self.B_mL = np.array((187,212,250))
        self.L = self.L_o + self.L_r[0] + self.L_m[0]
        self.Lb_r = np.array((296438,-145998))
        self.Lb_m = np.array((55594, 53581,160434))
        self.Lb_o = np.array((-43467*5,-59609*5))
        self.N = 500
        self.x = np.linspace(0,self.L,self.N)
        self.x_r = np.linspace(0, self.L_r, self.N)
        self.x_m1 = np.linspace(self.L_r[0], self.L_r[0] + self.L_m[:2], self.N)
        self.x_m2 = np.linspace(self.L_r[0] + self.L_m[1], self.L_r[0] + self.L_m[1] + self.L_m[2], self.N)
        self.x_m = np.column_stack((self.x_m1, self.x_m2))
        self.x_o = np.linspace(self.L_r[0] + self.L_m[0], self.L, self.N)
        self.z_r = np.linspace(-self.H_r,0,self.N)
        self.z_o = np.linspace(-self.H_o,0,self.N)
        self.z_m = np.linspace(-self.H_m,0,self.N)
        self.B_rx = self.B_river*np.exp((self.x_r-self.x_r[0])/self.Lb_r)
        self.B_mx = self.B_middle*np.exp((self.x_m-self.x_m[0])/self.Lb_m)
        self.B_ox = self.B_ocean*np.exp((self.x_o-self.x_o[0])/self.Lb_o)
        self.g = 9.81
        self.cycles = 2
        self.N_t = 500
        self.t = np.linspace(0,2*np.pi,(self.N_t))
        
        self.eta_riverflow = eta_riverflow
        self.u_riverflow = u_riverflow
        
        self.Av_r = Av*self.H_r
        self.Av_m = Av*self.H_m
        self.Av_o = Av*self.H_o
        self.Sf_r = Sf
        self.Sf_m = Sf
        self.Sf_o = Sf
        
        self.eta_r = np.zeros((len(self.x_r),len(self.t)))
        self.u_r = np.zeros((len(self.x_r),len(self.t)))
        self.eta_m = np.zeros((len(self.x_m),len(self.t)))
        self.u_m = np.zeros((len(self.x_m),len(self.t)))
        self.eta_o = np.zeros((len(self.x_o),len(self.t)))
        self.u_o = np.zeros((len(self.x_o),len(self.t)))

    def M2(self, M4 = "no"):
        # Determine if we're calculating M4 based on M4 parameter
        if M4 == "yes":
            a_M4 = 2
            b_M4 = 4
            # Only set ZM2 if not already set externally
            if not hasattr(self, 'ZM2') or self.ZM2 is None:
                self.ZM2 = 0.20*np.exp(-2j*1.87)  # Default external M4
        else:
            a_M4 = 1
            b_M4 = 1
            # Only set ZM2 if not already set externally
            if not hasattr(self, 'ZM2') or self.ZM2 is None:
                self.ZM2 = 0.80*np.exp(-1j*0.95)  # Default M2
        
        self.gamma_r = np.sqrt(-1j * self.omega*a_M4 / self.Av_r)
        self.alpha_r = self.Sf_r/(self.Sf_r * np.cosh(self.gamma_r*self.H_r) + self.Av_r * self.gamma_r * np.sinh(self.gamma_r*self.H_r))
        self.k1_r = (-(1/self.Lb_r) + np.sqrt((1/self.Lb_r)**2 + b_M4*4*self.omega**2/(((self.alpha_r/self.gamma_r) * np.sinh(self.gamma_r*self.H_r)-self.H_r)*self.g))) / 2
        self.k2_r = (-(1/self.Lb_r) - np.sqrt((1/self.Lb_r)**2 + b_M4*4*self.omega**2/(((self.alpha_r/self.gamma_r) * np.sinh(self.gamma_r*self.H_r)-self.H_r)*self.g))) / 2
        beta_r = 1j*self.g/(self.gamma_r*self.omega*a_M4) * (self.gamma_r*self.H_r - self.alpha_r*np.sinh(self.gamma_r*self.H_r))
        C1_river = self.k2_r/(self.k2_r*np.exp(self.k1_r*self.L_r) - self.k1_r*np.exp(self.k2_r*self.L_r))
        C2_river = -self.k1_r/(self.k2_r*np.exp(self.k1_r*self.L_r) - self.k1_r*np.exp(self.k2_r*self.L_r))
        C_river = beta_r*(self.k1_r*C1_river*np.exp(self.k1_r*self.L_r) + self.k2_r*C2_river*np.exp(self.k2_r*self.L_r))
        

        O = self.x_o[0]
        self.gamma_o = np.sqrt(-1j * self.omega*a_M4/ self.Av_o)
        self.alpha_o = self.Sf_o/(self.Sf_o * np.cosh(self.gamma_o*self.H_o) + self.Av_o * self.gamma_o * np.sinh(self.gamma_o*self.H_o))
        self.k1_o = (-(1/self.Lb_o) + np.sqrt((1/self.Lb_o)**2 + b_M4*4*self.omega**2/(((self.alpha_o/self.gamma_o) * np.sinh(self.gamma_o*self.H_o)-self.H_o)*self.g))) / 2
        self.k2_o = (-(1/self.Lb_o) - np.sqrt((1/self.Lb_o)**2 + b_M4*4*self.omega**2/(((self.alpha_o/self.gamma_o) * np.sinh(self.gamma_o*self.H_o)-self.H_o)*self.g))) / 2
        beta_o = 1j*self.g/(self.gamma_o*self.omega*a_M4) * (self.gamma_o*self.H_o - self.alpha_o*np.sinh(self.gamma_o*self.H_o))
        A_o = -np.exp(self.k2_o*self.L-self.k2_o*O)/(np.exp(self.k1_o*self.L)-np.exp(self.k1_o*O-self.k2_o*O+self.k2_o*self.L))
        B_o = self.ZM2/(np.exp(self.k1_o*self.L)-np.exp(self.k1_o*O-self.k2_o*O+self.k2_o*self.L))
        C_o = (1 - A_o*np.exp(self.k1_o*O))/np.exp(self.k2_o*O) 
        D_o = -B_o * np.exp(self.k1_o*O)/np.exp(self.k2_o*O)
        C1_ocean = beta_o*(self.k1_o*A_o*np.exp(self.k1_o*O) + self.k2_o*C_o*np.exp(self.k2_o*O))
        C2_ocean = beta_o*(self.k1_o*B_o*np.exp(self.k1_o*O) + self.k2_o*D_o*np.exp(self.k2_o*O))
        
        
        M = self.x_m[-1]
        R = self.x_m[0]
        self.gamma_m = np.sqrt(-1j * self.omega*a_M4 / self.Av_m)
        self.alpha_m = self.Sf_m/(self.Sf_m * np.cosh(self.gamma_m*self.H_m) + self.Av_m * self.gamma_m * np.sinh(self.gamma_m*self.H_m))
        self.k1_m = (-(1/self.Lb_m) + np.sqrt((1/self.Lb_m)**2 + b_M4*4*self.omega**2/(((self.alpha_m/self.gamma_m) * np.sinh(self.gamma_m*self.H_m)-self.H_m)*self.g))) / 2
        self.k2_m = (-(1/self.Lb_m) - np.sqrt((1/self.Lb_m)**2 + b_M4*4*self.omega**2/(((self.alpha_m/self.gamma_m) * np.sinh(self.gamma_m*self.H_m)-self.H_m)*self.g))) / 2
        beta_m = 1j*self.g/(self.gamma_m*self.omega*a_M4) * (self.gamma_m*self.H_m - self.alpha_m*np.sinh(self.gamma_m*self.H_m))
        B_m = 1/(np.exp(self.k1_m*(R + self.L_m)) - np.exp(self.k1_m*R+self.k2_m*(R + self.L_m)-self.k2_m*R))
        A_m = - (np.exp(self.k2_m*(R + self.L_m)-self.k2_m*R))/(np.exp(self.k1_m*(R + self.L_m)) - np.exp(self.k1_m*R+self.k2_m*(R + self.L_m)-self.k2_m*R))
        C_m = (1-A_m * np.exp(self.k1_m*R)) / np.exp(self.k2_m*R)
        D_m = - B_m * np.exp(self.k1_m*R) / np.exp(self.k2_m*R)
        C1_middle = beta_m * (A_m*self.k1_m * np.exp(self.k1_m*R) + C_m*self.k2_m * np.exp(self.k2_m*R))
        D1_middle =  beta_m * (B_m*self.k1_m * np.exp(self.k1_m*R) + D_m*self.k2_m * np.exp(self.k2_m*R))
        
        C2_middle = beta_m * (A_m*self.k1_m * np.exp(self.k1_m*M) + C_m*self.k2_m * np.exp(self.k2_m*M))
        D2_middle =  beta_m * (B_m*self.k1_m * np.exp(self.k1_m*M) + D_m*self.k2_m * np.exp(self.k2_m*M))
        
        C_v1 = np.sum(self.B_rx[-1,0] * C_river[0]) - np.sum(self.B_mx[0,0] * C1_middle[0]) - np.sum(self.B_mx[0,1] * C1_middle[1])
        D_v1 = -np.sum(self.B_mx[0,1] * D1_middle[1])
        E_v1 = -np.sum(self.B_mx[0,0] * D1_middle[0])
        F_v1 = 0
        C_v2 = np.sum(self.B_mx[-1,1] * C2_middle[1]) 
        D_v2 = np.sum(self.B_mx[-1,1] * D2_middle[1]) - np.sum(self.B_mx[0,2] * C1_middle[2]) + np.sum(self.B_rx[-1,1] * C_river[1]) #- np.sum(self.B_rx[1,-1] * C_river[1])
        E_v2 = -np.sum(self.B_mx[0,2]*D1_middle[2])
        F_v2 = 0
        C_v3 = np.sum(self.B_mx[-1,0] * C2_middle[0]) 
        D_v3 = np.sum(self.B_mx[-1,2] * C2_middle[2])
        E_v3 = -np.sum(self.B_ox[0,:]*C1_ocean) + np.sum(self.B_mx[-1,0] * D2_middle[0]) + np.sum(self.B_mx[-1,2] * D2_middle[2])
        F_v3 = np.sum(self.B_ox[0,:]*C2_ocean)
        
        a = np.array([[C_v1, D_v1, E_v1], [C_v2, D_v2, E_v2], [C_v3, D_v3, E_v3]])
        b = np.array([F_v1, F_v2, F_v3])
        
        self.eta_vertex_1 = np.linalg.solve(a,b)[0]
        self.eta_vertex_2 = np.linalg.solve(a,b)[1]
        self.eta_vertex_3 = np.linalg.solve(a,b)[2]
        
        C1_r1 = self.eta_vertex_1 * C1_river[0]
        C2_r1 = self.eta_vertex_1 * C2_river[0]
        C1_r2 = self.eta_vertex_2 * C1_river[1]
        C2_r2 = self.eta_vertex_2 * C2_river[1]
        C1_r = np.column_stack((C1_r1, C1_r2))[0]
        C2_r = np.column_stack((C2_r1, C2_r2))[0]
        C1_o1 = self.eta_vertex_3 * A_o[0] + B_o[0]  
        C2_o1 = self.eta_vertex_3 * C_o[0] + D_o[0]
        C1_o2 = self.eta_vertex_3 * A_o[1] + B_o[1]  
        C2_o2 = self.eta_vertex_3 * C_o[1] + D_o[1]
        C1_o = np.column_stack((C1_o1, C1_o2))[0]
        C2_o = np.column_stack((C2_o1, C2_o2))[0]

        C1_m1 = self.eta_vertex_1*A_m[0] + self.eta_vertex_3*B_m[0]
        C2_m1 = self.eta_vertex_1*C_m[0] + self.eta_vertex_3*D_m[0]
        C1_m2 = self.eta_vertex_1*A_m[1] + self.eta_vertex_2*B_m[1]
        C2_m2 = self.eta_vertex_1*C_m[1] + self.eta_vertex_2*D_m[1]
        C1_m3 = self.eta_vertex_2*A_m[2] + self.eta_vertex_3*B_m[2]
        C2_m3 = self.eta_vertex_2*C_m[2] + self.eta_vertex_3*D_m[2]
        C1_m = np.column_stack((C1_m1, C1_m2, C1_m3))[0]
        C2_m = np.column_stack((C2_m1, C2_m2, C2_m3))[0]
        
      
        self.xx_r, self.zz_r = np.zeros((self.N,len(self.x_r), len(self.H_r))), np.zeros((self.N,len(self.x_r), len(self.H_r)))       
        self.eta0_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        self.deta0dx_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        self.ddeta0ddx_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        self.u0_r = np.zeros((len(self.z_r),len(self.x_r), len(self.H_r)), dtype = complex)
        self.du0dz_r =  np.zeros((len(self.z_r),len(self.x_r), len(self.H_r)), dtype = complex)
        self.du0dx_r =  np.zeros((len(self.z_r),len(self.x_r), len(self.H_r)), dtype = complex)
        self.ddu0ddz_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        self.w0_r =  np.zeros((len(self.z_r),len(self.x_r), len(self.H_r)), dtype = complex)
        self.dw0dz_r =  np.zeros((len(self.z_r),len(self.x_r), len(self.H_r)), dtype = complex)
        for i in range(len(self.eta0_r[1])):
            self.xx_r[:,:,i], self.zz_r[:,:,i] = np.meshgrid(self.x_r[:,i], self.z_r[:,i])
            self.eta0_r[:,i] = C1_r[i] * np.exp(self.k1_r[i]*self.x_r[:,i]) + C2_r[i] * np.exp(self.k2_r[i]*self.x_r[:,i])  
            self.deta0dx_r[:,i] = C1_r[i] * self.k1_r[i] * np.exp(self.k1_r[i]*self.x_r[:,i]) + C2_r[i] * self.k2_r[i] * np.exp(self.k2_r[i]*self.x_r[:,i])
            self.ddeta0ddx_r[:,i] = C1_r[i] * self.k1_r[i]**2 * np.exp(self.k1_r[i]*self.x_r[:,i]) + C2_r[i] * self.k2_r[i]**2 * np.exp(self.k2_r[i]*self.x_r[:,i])

            self.u0_r[:,:,i] = 1j*self.g/(self.omega*a_M4) * (self.alpha_r[i] * np.cosh(self.gamma_r[i]*self.zz_r[:,:,i])-1) * self.deta0dx_r[:,i]
            self.du0dz_r[:,:,i] = 1j*self.g/(self.omega*a_M4) * (self.alpha_r[i]*self.gamma_r[i] * np.sinh(self.gamma_r[i]*self.zz_r[:,:,i])) * self.deta0dx_r[:,i]
            self.du0dx_r[:,:,i] = 1j*self.g/(self.omega*a_M4) * (self.alpha_r[i] * np.cosh(self.gamma_r[i]*self.zz_r[:,:,i])-1) * self.ddeta0ddx_r[:,i]
            self.ddu0ddz_r[:,i] = 1j*self.g*self.alpha_r[i]*self.gamma_r[i]**2/(self.omega*a_M4) * np.cosh(0) * self.deta0dx_r[:,i]
            
            self.w0_r[:,:,i] = -1j*self.g/(self.omega*a_M4) * (self.alpha_r[i]/self.gamma_r[i] * np.sinh(self.gamma_r[i]*self.zz_r[:,:,i]) - self.zz_r[:,:,i]) * \
                        (self.ddeta0ddx_r[:,i] + 1/self.Lb_r[i] * self.deta0dx_r[:,i]) - 1j*self.omega*self.eta0_r[:,i]
            self.dw0dz_r[:,:,i] = -1j*self.g/(self.omega*a_M4) * (self.alpha_r[i] * np.cosh(self.gamma_r[i]*self.zz_r[:,:,i]) - 1) * \
                        (self.ddeta0ddx_r[:,i] + 1/self.Lb_r[i] * self.deta0dx_r[:,i])

         
        self.xx_o, self.zz_o = np.zeros((self.N,len(self.x_o), len(self.H_o))), np.zeros((self.N,len(self.x_o), len(self.H_o)))       
        self.eta0_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        self.deta0dx_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        self.ddeta0ddx_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        self.u0_o = np.zeros((len(self.z_o),len(self.x_o), len(self.H_o)), dtype = complex)
        self.du0dz_o =  np.zeros((len(self.z_o),len(self.x_o), len(self.H_o)), dtype = complex)
        self.du0dx_o =  np.zeros((len(self.z_o),len(self.x_o), len(self.H_o)), dtype = complex)
        self.ddu0ddz_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        self.w0_o =  np.zeros((len(self.z_o),len(self.x_o), len(self.H_o)), dtype = complex)
        self.dw0dz_o =  np.zeros((len(self.z_o),len(self.x_o), len(self.H_o)), dtype = complex)
        for i in range(len(self.eta0_o[1])):
            self.xx_o[:,:,i], self.zz_o[:,:,i] = np.meshgrid(self.x_o[:,i], self.z_o[:,i])
            self.eta0_o[:,i] = C1_o[i] * np.exp(self.k1_o[i]*self.x_o[:,i]) + C2_o[i] * np.exp(self.k2_o[i]*self.x_o[:,i])  
            self.deta0dx_o[:,i] = C1_o[i] * self.k1_o[i] * np.exp(self.k1_o[i]*self.x_o[:,i]) + C2_o[i] * self.k2_o[i] * np.exp(self.k2_o[i]*self.x_o[:,i])
            self.ddeta0ddx_o[:,i] = C1_o[i] * self.k1_o[i]**2 * np.exp(self.k1_o[i]*self.x_o[:,i]) + C2_o[i] * self.k2_o[i]**2 * np.exp(self.k2_o[i]*self.x_o[:,i])

            self.u0_o[:,:,i] = 1j*self.g/(self.omega*a_M4) * (self.alpha_o[i] * np.cosh(self.gamma_o[i]*self.zz_o[:,:,i])-1) * self.deta0dx_o[:,i]
            self.du0dz_o[:,:,i] = 1j*self.g/(self.omega*a_M4) * (self.alpha_o[i]*self.gamma_o[i] * np.sinh(self.gamma_o[i]*self.zz_o[:,:,i])) * self.deta0dx_o[:,i]
            self.du0dx_o[:,:,i] = 1j*self.g/(self.omega*a_M4) * (self.alpha_o[i] * np.cosh(self.gamma_o[i]*self.zz_o[:,:,i])-1) * self.ddeta0ddx_o[:,i]
            self.ddu0ddz_o[:,i] = 1j*self.g*self.alpha_o[i]*self.gamma_o[i]**2/(self.omega*a_M4) * np.cosh(0) * self.deta0dx_o[:,i]
            
            self.w0_o[:,:,i] = -1j*self.g/(self.omega*a_M4) * (self.alpha_o[i]/self.gamma_o[i] * np.sinh(self.gamma_o[i]*self.zz_o[:,:,i]) - self.zz_o[:,:,i]) * \
                        (self.ddeta0ddx_o[:,i] + 1/self.Lb_o[i] * self.deta0dx_o[:,i]) - 1j*self.omega*self.eta0_o[:,i]
            self.dw0dz_o[:,:,i] = -1j*self.g/(self.omega*a_M4) * (self.alpha_o[i] * np.cosh(self.gamma_o[i]*self.zz_o[:,:,i]) - 1) * \
                        (self.ddeta0ddx_o[:,i] + 1/self.Lb_o[i] * self.deta0dx_o[:,i])

  
        self.xx_m, self.zz_m = np.zeros((self.N,len(self.x_m), len(self.H_m))), np.zeros((self.N,len(self.x_m), len(self.H_m)))       
        self.eta0_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        self.deta0dx_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        self.ddeta0ddx_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        self.u0_m = np.zeros((len(self.z_m),len(self.x_m), len(self.H_m)), dtype = complex)
        self.du0dz_m =  np.zeros((len(self.z_m),len(self.x_m), len(self.H_m)), dtype = complex)
        self.du0dx_m =  np.zeros((len(self.z_m),len(self.x_m), len(self.H_m)), dtype = complex)
        self.ddu0ddz_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        self.w0_m =  np.zeros((len(self.z_m),len(self.x_m), len(self.H_m)), dtype = complex)
        self.dw0dz_m =  np.zeros((len(self.z_m),len(self.x_m), len(self.H_m)), dtype = complex)
        for i in range(len(self.eta0_m[1])):
            self.xx_m[:,:,i], self.zz_m[:,:,i] = np.meshgrid(self.x_m[:,i], self.z_m[:,i])
            self.eta0_m[:,i] = C1_m[i] * np.exp(self.k1_m[i]*self.x_m[:,i]) + C2_m[i] * np.exp(self.k2_m[i]*self.x_m[:,i])  
            self.deta0dx_m[:,i] = C1_m[i] * self.k1_m[i] * np.exp(self.k1_m[i]*self.x_m[:,i]) + C2_m[i] * self.k2_m[i] * np.exp(self.k2_m[i]*self.x_m[:,i])
            self.ddeta0ddx_m[:,i] = C1_m[i] * self.k1_m[i]**2 * np.exp(self.k1_m[i]*self.x_m[:,i]) + C2_m[i] * self.k2_m[i]**2 * np.exp(self.k2_m[i]*self.x_m[:,i])

            self.u0_m[:,:,i] = 1j*self.g/(self.omega*a_M4) * (self.alpha_m[i] * np.cosh(self.gamma_m[i]*self.zz_m[:,:,i])-1) * self.deta0dx_m[:,i]
            self.du0dz_m[:,:,i] = 1j*self.g/(self.omega*a_M4) * (self.alpha_m[i]*self.gamma_m[i] * np.sinh(self.gamma_m[i]*self.zz_m[:,:,i])) * self.deta0dx_m[:,i]
            self.du0dx_m[:,:,i] = 1j*self.g/(self.omega*a_M4) * (self.alpha_m[i] * np.cosh(self.gamma_m[i]*self.zz_m[:,:,i])-1) * self.ddeta0ddx_m[:,i]
            self.ddu0ddz_m[:,i] = 1j*self.g*self.alpha_m[i]*self.gamma_m[i]**2/(self.omega*a_M4) * np.cosh(0) * self.deta0dx_m[:,i]
            
            self.w0_m[:,:,i] = -1j*self.g/(self.omega*a_M4) * (self.alpha_m[i]/self.gamma_m[i] * np.sinh(self.gamma_m[i]*self.zz_m[:,:,i]) - self.zz_m[:,:,i]) * \
                        (self.ddeta0ddx_m[:,i] + 1/self.Lb_m[i] * self.deta0dx_m[:,i]) - 1j*self.omega*self.eta0_m[:,i]
            self.dw0dz_m[:,:,i] = -1j*self.g/(self.omega*a_M4) * (self.alpha_m[i] * np.cosh(self.gamma_m[i]*self.zz_m[:,:,i]) - 1) * \
                        (self.ddeta0ddx_m[:,i] + 1/self.Lb_m[i] * self.deta0dx_m[:,i])

        self.u0_mean_r = np.mean(self.u0_r, axis = 0)
        self.u0_mean_m = np.mean(self.u0_m, axis = 0)
        self.u0_mean_o = np.mean(self.u0_o, axis = 0)
            
        
    def M4_ext(self):
        self.M2(M4 = "yes")
        self.eta14_ext_r = self.eta0_r
        self.eta14_ext_m = self.eta0_m
        self.eta14_ext_o = self.eta0_o
        
        self.u14_ext_r = self.u0_mean_r
        self.u14_ext_m = self.u0_mean_m
        self.u14_ext_o = self.u0_mean_o
        
        self.M2()
        
    def M4(self):        
        self.M2(M4 = "no")
        self.gammaM4_r = np.sqrt(-2j * self.omega / self.Av_r)
        self.alphaM4_r = 1/ (np.cosh(self.gammaM4_r * self.H_r) + self.Av_r * self.gammaM4_r * np.sinh(self.gammaM4_r * self.H_r)/ self.Sf_r) 
        self.p1_r = (-(1/self.Lb_r) + np.sqrt((1/self.Lb_r)**2 + 16*self.omega**2/(((self.alphaM4_r/self.gammaM4_r) * np.sinh(self.gammaM4_r*self.H_r)-self.H_r)*self.g))) / 2
        self.p2_r = (-(1/self.Lb_r) - np.sqrt((1/self.Lb_r)**2 + 16*self.omega**2/(((self.alphaM4_r/self.gammaM4_r) * np.sinh(self.gammaM4_r*self.H_r)-self.H_r)*self.g))) / 2
        self.W_r  = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        for i in range(len(self.W_r[1])):
            self.W_r[:,i] = (self.p2_r[i]-self.p1_r[i]) * np.exp((self.p2_r[i]+self.p1_r[i]) * self.x_r[:,i])
            
        self.gammaM4_m = np.sqrt(-2j * self.omega / self.Av_m)
        self.alphaM4_m = 1/ (np.cosh(self.gammaM4_m * self.H_m) + self.Av_m * self.gammaM4_m * np.sinh(self.gammaM4_m * self.H_m)/ self.Sf_m) 
        self.p1_m = (-(1/self.Lb_m) + np.sqrt((1/self.Lb_m)**2 + 16*self.omega**2/(((self.alphaM4_m/self.gammaM4_m) * np.sinh(self.gammaM4_m*self.H_m)-self.H_m)*self.g))) / 2
        self.p2_m = (-(1/self.Lb_m) - np.sqrt((1/self.Lb_m)**2 + 16*self.omega**2/(((self.alphaM4_m/self.gammaM4_m) * np.sinh(self.gammaM4_m*self.H_m)-self.H_m)*self.g))) / 2
        self.W_m  = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        for i in range(len(self.W_m[1])):
            self.W_m[:,i] = (self.p2_m[i]-self.p1_m[i]) * np.exp((self.p2_m[i]+self.p1_m[i]) * self.x_m[:,i])

        self.gammaM4_o = np.sqrt(-2j * self.omega / self.Av_o)
        self.alphaM4_o = 1/ (np.cosh(self.gammaM4_o * self.H_o) + self.Av_o * self.gammaM4_o * np.sinh(self.gammaM4_o * self.H_o)/ self.Sf_o) 
        self.p1_o = (-(1/self.Lb_o) + np.sqrt((1/self.Lb_o)**2 + 16*self.omega**2/(((self.alphaM4_o/self.gammaM4_o) * np.sinh(self.gammaM4_o*self.H_o)-self.H_o)*self.g))) / 2
        self.p2_o = (-(1/self.Lb_o) - np.sqrt((1/self.Lb_o)**2 + 16*self.omega**2/(((self.alphaM4_o/self.gammaM4_o) * np.sinh(self.gammaM4_o*self.H_o)-self.H_o)*self.g))) / 2
        self.W_o  = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        for i in range(len(self.W_o[1])):
            self.W_o[:,i] = (self.p2_o[i]-self.p1_o[i]) * np.exp((self.p2_o[i]+self.p1_o[i]) * self.x_o[:,i])

        self.C_M4_r = self.alphaM4_r / self.gammaM4_r * np.sinh(self.gammaM4_r*self.H_r) - self.H_r
        self.C_M4_m = self.alphaM4_m / self.gammaM4_m * np.sinh(self.gammaM4_m*self.H_m) - self.H_m
        self.C_M4_o = self.alphaM4_o / self.gammaM4_o * np.sinh(self.gammaM4_o*self.H_o) - self.H_o         
        
    def M4_total(self):
        self.M4_ext()
        self.eta14_r = self.eta14_no_stress_r + self.eta14_stokes_r + self.eta14_adv_r 
        self.eta14_m = self.eta14_no_stress_m + self.eta14_stokes_m + self.eta14_adv_m 
        self.eta14_o = self.eta14_no_stress_o + self.eta14_stokes_o + self.eta14_adv_o 
        
        self.eta14_r_t = self.eta14_no_stress_r + self.eta14_stokes_r + self.eta14_adv_r + self.eta14_ext_r
        self.eta14_m_t = self.eta14_no_stress_m + self.eta14_stokes_m + self.eta14_adv_m + self.eta14_ext_m
        self.eta14_o_t = self.eta14_no_stress_o + self.eta14_stokes_o + self.eta14_adv_o + self.eta14_ext_o
        
        self.u14_r = self.u14_no_stress_r + self.u14_stokes_r + self.u14_adv_r 
        self.u14_m = self.u14_no_stress_m + self.u14_stokes_m + self.u14_adv_m
        self.u14_o = self.u14_no_stress_o + self.u14_stokes_o + self.u14_adv_o
        
        self.M2(M4 = "yes")
        self.u14_m_AS = self.u14_no_stress_m[:,:,0] + self.u14_stokes_m[:,:,0] + self.u14_adv_m[:,:,0] + self.u0_m[:,:,0]
        self.u14_o_AS = self.u14_no_stress_o[:,:,0] + self.u14_stokes_o[:,:,0] + self.u14_adv_o[:,:,0] + self.u0_o[:,:,0]

        self.M2()
        
        self.u14_mean_r = np.mean(self.u14_r, axis = 0) + self.u14_ext_r
        self.u14_mean_m = np.mean(self.u14_m, axis = 0) + self.u14_ext_m
        self.u14_mean_o = np.mean(self.u14_o, axis = 0) + self.u14_ext_o
        
    def full_tide(self, first_order = "no"):
        if first_order == "no":
            self.eta_r = np.zeros((len(self.z_r),len(self.t), len(self.H_r)))
            self.u_r = np.zeros((len(self.z_r),len(self.t), len(self.H_r)))
            for j in range(len(self.H_r)):
                for i in range(len(self.x_r)):
                    self.eta_r[i,:,j] = abs(self.eta0_r)[i,j] * np.cos(self.t-np.angle(self.eta0_r[i,j])) 
                    self.u_r[i,:,j] = abs(self.u0_rean_r)[i,j] * np.cos(self.t-np.angle(self.u0_rean_r[i,j])) 
                
            self.eta_m = np.zeros((len(self.z_m),len(self.t), len(self.H_m)))
            self.u_m = np.zeros((len(self.z_m),len(self.t), len(self.H_m)))
            for j in range(len(self.H_m)):
                for i in range(len(self.x_m)):
                    self.eta_m[i,:,j] = abs(self.eta0_m)[i,j] * np.cos(self.t-np.angle(self.eta0_m[i,j])) 
                    self.u_m[i,:,j] = abs(self.u0_mean_m)[i,j] * np.cos(self.t-np.angle(self.u0_mean_m[i,j])) 

            self.eta_o = np.zeros((len(self.z_o),len(self.t), len(self.H_o)))
            self.u_o = np.zeros((len(self.z_o),len(self.t), len(self.H_o)))
            for j in range(len(self.H_o)):
                for i in range(len(self.x_o)):
                    self.eta_o[i,:,j] = abs(self.eta0_o)[i,j] * np.cos(self.t-np.angle(self.eta0_o[i,j])) 
                    self.u_o[i,:,j] = abs(self.u0_mean_o)[i,j] * np.cos(self.t-np.angle(self.u0_mean_o[i,j])) 

                
        elif first_order == "yes": 
            # self.eta_riverflow_r = self.eta_riverflow[:,5:]
            # self.u_riverflow_r = self.u_riverflow[:,5:]
    
            self.eta_riverflow_r = 0
            self.u_riverflow_r = 0
    
            self.eta_r = np.zeros((len(self.z_r),len(self.t), len(self.H_r)))
            self.u_r = np.zeros((len(self.z_r),len(self.t), len(self.H_r)))
            for j in range(len(self.H_r)):
                for i in range(len(self.x_r)):
                    self.eta_r[i,:,j] = abs(self.eta0_r)[i,j] * np.cos(self.t-np.angle(self.eta0_r[i,j])) \
                        + abs(self.eta14_r_t)[i,j] * np.cos(2*(self.t-np.angle(self.eta14_r[i,j]))) 
                        #+ self.eta_riverflow_r[i,j]
                    self.u_r[i,:,j] = abs(self.u0_mean_r[i,j]) * np.cos(self.t-np.angle(self.u0_mean_r[i,j])) \
                        + abs(self.u14_mean_r[i,j]) * np.cos(2*(self.t-np.angle(self.u14_mean_r[i,j]))) 
                        #+ self.u_riverflow_r[i,j]
 
            #self.eta_riverflow_m = self.eta_riverflow[:,2:5]
            #self.u_riverflow_m = self.u_riverflow[:,2:5]
            
            self.eta_riverflow_m = 0
            self.u_riverflow_m = 0
            
            self.eta_m = np.zeros((len(self.z_m),len(self.t), len(self.H_m)))
            self.u_m = np.zeros((len(self.z_m),len(self.t), len(self.H_m)))
            for j in range(len(self.H_m)):
                for i in range(len(self.x_m)):
                    self.eta_m[i,:,j] = abs(self.eta0_m)[i,j] * np.cos(self.t-np.angle(self.eta0_m[i,j])) \
                        + abs(self.eta14_m_t)[i,j] * np.cos(2*(self.t-np.angle(self.eta14_m[i,j]))) 
                        #+ self.eta_riverflow_m[i,j]
                    self.u_m[i,:,j] = abs(self.u0_mean_m[i,j]) * np.cos(self.t-np.angle(self.u0_mean_m[i,j])) \
                        + abs(self.u14_mean_m[i,j]) * np.cos(2*(self.t-np.angle(self.u14_mean_m[i,j]))) 
                        #+ self.u_riverflow_m[i,j]
  
    
            #self.eta_riverflow_o = self.eta_riverflow[:,:2]
            #self.u_riverflow_o = self.u_riverflow[:,:2]
            
            self.eta_riverflow_o = 0
            self.u_riverflow_o = 0
            
            self.eta_o = np.zeros((len(self.z_o),len(self.t), len(self.H_o)))
            self.u_o = np.zeros((len(self.z_o),len(self.t), len(self.H_o)))
            for j in range(len(self.H_o)):
                for i in range(len(self.x_o)):
                    self.eta_o[i,:,j] = abs(self.eta0_o)[i,j] * np.cos(self.t-np.angle(self.eta0_o[i,j])) \
                        + abs(self.eta14_o_t)[i,j] * np.cos(2*(self.t-np.angle(self.eta14_o[i,j]))) 
                        #+ self.eta_riverflow_o[i,j]
                    self.u_o[i,:,j] = abs(self.u0_mean_o[i,j]) * np.cos(self.t-np.angle(self.u0_mean_o[i,j])) \
                        + abs(self.u14_mean_o[i,j]) * np.cos(2*(self.t-np.angle(self.u14_mean_o[i,j]))) 
                        #+ self.u_riverflow_o[i,j]

            # self.bottom = np.zeros((self.N, self.N))
            # for i in range(self.N):
            #         self.bottom[i,:] = abs(self.u0_m[-1,:,0]) * np.cos(self.t-np.angle(self.u0_m[-1,:,0])) \
            #             + abs(self.u14_m_AS[i]) * np.cos(2*self.t-np.angle(self.u14_m_AS[i]))

    def stokes_M4(self): 
        betaM4_r = -1j*self.g/(self.gammaM4_r*self.omega) * (self.gammaM4_r*self.H_r - self.alphaM4_r*np.sinh(self.gammaM4_r*self.H_r))
        betaM4_m = -1j*self.g/(self.gammaM4_m*self.omega) * (self.gammaM4_m*self.H_m - self.alphaM4_m*np.sinh(self.gammaM4_m*self.H_m))
        betaM4_o = -1j*self.g/(self.gammaM4_o*self.omega) * (self.gammaM4_o*self.H_o - self.alphaM4_o*np.sinh(self.gammaM4_o*self.H_o))
        coefM4_r = -1j * self.g / (2*self.omega) * (self.H_r - self.alphaM4_r / self.gammaM4_r * np.sinh(self.gammaM4_r*self.H_r)) 
        coefM4_m = -1j * self.g / (2*self.omega) * (self.H_m - self.alphaM4_m / self.gammaM4_m * np.sinh(self.gammaM4_m*self.H_m)) 
        coefM4_o = -1j * self.g / (2*self.omega) * (self.H_o - self.alphaM4_o / self.gammaM4_o * np.sinh(self.gammaM4_o*self.H_o)) 
        CM4_r = np.repeat(0, len(self.x_r))
        CM4_m = np.repeat(0, len(self.x_m))
        CM4_o = np.repeat(0, len(self.x_o))

        F_14_stokes_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        detaudx_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        F_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        for i in range(len(self.eta0_r[1])):
            F_14_stokes_r[:,i] = 1/2 * self.eta0_r[:,i] * self.u0_r[-1,:,i]        
            detaudx_r[:,i] = np.gradient(F_14_stokes_r[:,i], self.x_r[:,i], edge_order = 2, axis = 0)
            F_r[:,i] = -(detaudx_r[:,i] + F_14_stokes_r[:,i]/self.Lb_r[i]) / coefM4_r[i]  
            
        F_14_stokes_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        detaudx_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        F_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        for i in range(len(self.eta0_m[1])):
            F_14_stokes_m[:,i] = 1/2 * self.eta0_m[:,i] * self.u0_m[-1,:,i]        
            detaudx_m[:,i] = np.gradient(F_14_stokes_m[:,i], self.x_m[:,i], edge_order = 2, axis = 0)
            F_m[:,i] = -(detaudx_m[:,i] + F_14_stokes_m[:,i]/self.Lb_m[i]) / coefM4_m[i] 

        F_14_stokes_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        detaudx_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        F_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        for i in range(len(self.eta0_o[1])):
            F_14_stokes_o[:,i] = 1/2 * self.eta0_o[:,i] * self.u0_o[-1,:,i]        
            detaudx_o[:,i] = np.gradient(F_14_stokes_o[:,i], self.x_o[:,i], edge_order = 2, axis = 0)
            F_o[:,i] = -(detaudx_o[:,i] + F_14_stokes_o[:,i]/self.Lb_o[i]) / coefM4_o[i] 
        
        self.matching(F_r, betaM4_r, CM4_r, "river")
        self.matching(F_m, betaM4_m, CM4_m, "middle")
        self.matching(F_o, betaM4_o, CM4_o, "ocean")
        self.vertex()
        self.eta14_stokes_r, self.eta14_stokes_m, self.eta14_stokes_o = self.eta14()
        
        deta_14_stokes_dx_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        self.u14_stokes_r = np.zeros((len(self.z_r),len(self.x_r), len(self.H_r)), dtype = complex)
        for i in range(len(self.eta0_r[1])):
            deta_14_stokes_dx_r[:,i] = self.C1_r[i]*self.p1_r[i]*np.exp(self.p1_r[i]*self.x_r[:,i]) + self.C2_r[i]*self.p2_r[i]*np.exp(self.p2_r[i]*self.x_r[:,i]) + self.Ar[:,i]*self.p1_r[i]*np.exp(self.p1_r[i]*self.x_r[:,i]) + self.Br[:,i]*self.p2_r[i]*np.exp(self.p2_r[i]*self.x_r[:,i]) + self.Ax_r[:,i]*np.exp(self.p1_r[i]*self.x_r[:,i]) + self.Bx_r[:,i]*np.exp(self.p2_r[i]*self.x_r[:,i])
            self.u14_stokes_r[:,:,i] =  -1j*self.g/(2*self.omega) * (1-self.alphaM4_r[i]*np.cosh(self.gammaM4_r[i]*self.zz_r[:,:,i])) *  deta_14_stokes_dx_r[:,i]
        
        deta_14_stokes_dx_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        self.u14_stokes_m = np.zeros((len(self.z_m),len(self.x_m), len(self.H_m)), dtype = complex)
        for i in range(len(self.eta0_m[1])):
            deta_14_stokes_dx_m[:,i] = self.C1_m[i]*self.p1_m[i]*np.exp(self.p1_m[i]*self.x_m[:,i]) + self.C2_m[i]*self.p2_m[i]*np.exp(self.p2_m[i]*self.x_m[:,i]) + self.Am[:,i]*self.p1_m[i]*np.exp(self.p1_m[i]*self.x_m[:,i]) + self.Bm[:,i]*self.p2_m[i]*np.exp(self.p2_m[i]*self.x_m[:,i]) + self.Ax_m[:,i]*np.exp(self.p1_m[i]*self.x_m[:,i]) + self.Bx_m[:,i]*np.exp(self.p2_m[i]*self.x_m[:,i])
            self.u14_stokes_m[:,:,i] =  -1j*self.g/(2*self.omega) * (1-self.alphaM4_m[i]*np.cosh(self.gammaM4_m[i]*self.zz_m[:,:,i])) *  deta_14_stokes_dx_m[:,i]


        deta_14_stokes_dx_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        self.u14_stokes_o = np.zeros((len(self.z_o),len(self.x_o), len(self.H_o)), dtype = complex)
        for i in range(len(self.eta0_o[1])):
            deta_14_stokes_dx_o[:,i] = self.C1_o[i]*self.p1_o[i]*np.exp(self.p1_o[i]*self.x_o[:,i]) + self.C2_o[i]*self.p2_o[i]*np.exp(self.p2_o[i]*self.x_o[:,i]) + self.Ao[:,i]*self.p1_o[i]*np.exp(self.p1_o[i]*self.x_o[:,i]) + self.Bo[:,i]*self.p2_o[i]*np.exp(self.p2_o[i]*self.x_o[:,i]) + self.Ax_o[:,i]*np.exp(self.p1_o[i]*self.x_o[:,i]) + self.Bx_o[:,i]*np.exp(self.p2_o[i]*self.x_o[:,i])
            self.u14_stokes_o[:,:,i] =  -1j*self.g/(2*self.omega) * (1-self.alphaM4_o[i]*np.cosh(self.gammaM4_o[i]*self.zz_o[:,:,i])) *  deta_14_stokes_dx_o[:,i]


    def no_stress_M4(self):
        betaM4_r = 1j*self.g/(self.gammaM4_r*self.omega) * (self.gammaM4_r*self.H_r - self.alphaM4_r*np.sinh(self.gammaM4_r*self.H_r))
        betaM4_m = 1j*self.g/(self.gammaM4_m*self.omega) * (self.gammaM4_m*self.H_m - self.alphaM4_m*np.sinh(self.gammaM4_m*self.H_m))
        betaM4_o = 1j*self.g/(self.gammaM4_o*self.omega) * (self.gammaM4_o*self.H_o - self.alphaM4_o*np.sinh(self.gammaM4_o*self.H_o))

        F_14_no_stress_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        F_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        CM4_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        for i in range(len(self.eta0_r[1])):
            F_14_no_stress_r[:,i] = 1/2 * (self.ddu0ddz_r[:,i] * self.eta0_r[:,i])        
            F_r[:,i] = (2j*self.omega/self.g*((1/self.Lb_r[i] * F_14_no_stress_r[:,i] + np.gradient(F_14_no_stress_r[:,i], self.x_r[:,i], edge_order = 2)) * (1-self.alphaM4_r[i])/self.gammaM4_r[i]**2))/self.C_M4_r[i]
            CM4_r[:,i] = np.trapz((-self.alphaM4_r[i]/self.gammaM4_r[i]*F_14_no_stress_r[:,i] * (np.sinh(self.gammaM4_r[i]*(self.zz_r[:,:,i]+self.H_r[i])) + self.Av_r[i]*self.gammaM4_r[i]/self.Sf_r*np.cosh(self.gammaM4_r[i]*(self.zz_r[:,:,i]+self.H_r[i])))), x=self.z_r[:,i], axis = 0)
        
        F_14_no_stress_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        F_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        CM4_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        for i in range(len(self.eta0_m[1])):
            F_14_no_stress_m[:,i] = 1/2 * (self.ddu0ddz_m[:,i] * self.eta0_m[:,i])        
            F_m[:,i] = (2j*self.omega/self.g*((1/self.Lb_m[i] * F_14_no_stress_m[:,i] + np.gradient(F_14_no_stress_m[:,i], self.x_m[:,i], edge_order = 2)) * (1-self.alphaM4_m[i])/self.gammaM4_m[i]**2))/self.C_M4_m[i]
            CM4_m[:,i] = np.trapz((-self.alphaM4_m[i]/self.gammaM4_m[i]*F_14_no_stress_m[:,i] * (np.sinh(self.gammaM4_m[i]*(self.zz_m[:,:,i]+self.H_m[i])) + self.Av_m[i]*self.gammaM4_m[i]/self.Sf_m*np.cosh(self.gammaM4_m[i]*(self.zz_m[:,:,i]+self.H_m[i])))), x=self.z_m[:,i], axis = 0)

        F_14_no_stress_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        F_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        CM4_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        for i in range(len(self.eta0_o[1])):
            F_14_no_stress_o[:,i] = 1/2 * (self.ddu0ddz_o[:,i] * self.eta0_o[:,i])        
            F_o[:,i] = (2j*self.omega/self.g*((1/self.Lb_o[i] * F_14_no_stress_o[:,i] + np.gradient(F_14_no_stress_o[:,i], self.x_o[:,i], edge_order = 2)) * (1-self.alphaM4_o[i])/self.gammaM4_o[i]**2))/self.C_M4_o[i]
            CM4_o[:,i] = np.trapz((-self.alphaM4_o[i]/self.gammaM4_o[i]*F_14_no_stress_o[:,i] * (np.sinh(self.gammaM4_o[i]*(self.zz_o[:,:,i]+self.H_o[i])) + self.Av_o[i]*self.gammaM4_o[i]/self.Sf_r*np.cosh(self.gammaM4_o[i]*(self.zz_o[:,:,i]+self.H_o[i])))), x=self.z_o[:,i], axis = 0)

        self.matching(F_r, betaM4_r, CM4_r, "river")
        self.matching(F_m, betaM4_m, CM4_m, "middle")
        self.matching(F_o, betaM4_o, CM4_o, "ocean")
        self.vertex()
        self.eta14_no_stress_r, self.eta14_no_stress_m, self.eta14_no_stress_o = self.eta14()
        
        deta_14_no_stress_dx_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        self.u14_no_stress_r = np.zeros((len(self.z_r),len(self.x_r), len(self.H_r)), dtype = complex)
        for i in range(len(self.eta0_r[1])):
            deta_14_no_stress_dx_r[:,i] = self.C1_r[i]*self.p1_r[i]*np.exp(self.p1_r[i]*self.x_r[:,i]) + self.C2_r[i]*self.p2_r[i]*np.exp(self.p2_r[i]*self.x_r[:,i]) + self.Ar[:,i]*self.p1_r[i]*np.exp(self.p1_r[i]*self.x_r[:,i]) + self.Br[:,i]*self.p2_r[i]*np.exp(self.p2_r[i]*self.x_r[:,i]) + self.Ax_r[:,i]*np.exp(self.p1_r[i]*self.x_r[:,i]) + self.Bx_r[:,i]*np.exp(self.p2_r[i]*self.x_r[:,i])
            self.u14_no_stress_r[:,:,i] =  1j*self.g/(2*self.omega)*(1-self.alphaM4_r[i]*np.cosh(self.gammaM4_r[i] * self.zz_r[:,:,i])) *  deta_14_no_stress_dx_r[:,i] - self.alphaM4_r[i]/self.gammaM4_r[i]*F_14_no_stress_r[:,i] * (np.sinh(self.gammaM4_r[i]*(self.zz_r[:,:,i]+self.H_r[i])) + self.Av_r[i]*self.gammaM4_r[i]/self.Sf_r*np.cosh(self.gammaM4_r[i]*(self.zz_r[:,:,i]+self.H_r[i])))
     
        deta_14_no_stress_dx_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        self.u14_no_stress_m = np.zeros((len(self.z_m),len(self.x_m), len(self.H_m)), dtype = complex)
        for i in range(len(self.eta0_m[1])):
            deta_14_no_stress_dx_m[:,i] = self.C1_m[i]*self.p1_m[i]*np.exp(self.p1_m[i]*self.x_m[:,i]) + self.C2_m[i]*self.p2_m[i]*np.exp(self.p2_m[i]*self.x_m[:,i]) + self.Am[:,i]*self.p1_m[i]*np.exp(self.p1_m[i]*self.x_m[:,i]) + self.Bm[:,i]*self.p2_m[i]*np.exp(self.p2_m[i]*self.x_m[:,i]) + self.Ax_m[:,i]*np.exp(self.p1_m[i]*self.x_m[:,i]) + self.Bx_m[:,i]*np.exp(self.p2_m[i]*self.x_m[:,i])
            self.u14_no_stress_m[:,:,i] =  1j*self.g/(2*self.omega)*(1-self.alphaM4_m[i]*np.cosh(self.gammaM4_m[i] * self.zz_m[:,:,i])) *  deta_14_no_stress_dx_m[:,i] - self.alphaM4_m[i]/self.gammaM4_m[i]*F_14_no_stress_m[:,i] * (np.sinh(self.gammaM4_m[i]*(self.zz_m[:,:,i]+self.H_m[i])) + self.Av_m[i]*self.gammaM4_m[i]/self.Sf_m*np.cosh(self.gammaM4_m[i]*(self.zz_m[:,:,i]+self.H_m[i])))

            
        deta_14_no_stress_dx_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        self.u14_no_stress_o = np.zeros((len(self.z_o),len(self.x_o), len(self.H_o)), dtype = complex)
        for i in range(len(self.eta0_o[1])):
            deta_14_no_stress_dx_o[:,i] = self.C1_o[i]*self.p1_o[i]*np.exp(self.p1_o[i]*self.x_o[:,i]) + self.C2_o[i]*self.p2_o[i]*np.exp(self.p2_o[i]*self.x_o[:,i]) + self.Ao[:,i]*self.p1_o[i]*np.exp(self.p1_o[i]*self.x_o[:,i]) + self.Bo[:,i]*self.p2_o[i]*np.exp(self.p2_o[i]*self.x_o[:,i]) + self.Ax_o[:,i]*np.exp(self.p1_o[i]*self.x_o[:,i]) + self.Bx_o[:,i]*np.exp(self.p2_o[i]*self.x_o[:,i])
            self.u14_no_stress_o[:,:,i] =  1j*self.g/(2*self.omega)*(1-self.alphaM4_o[i]*np.cosh(self.gammaM4_o[i] * self.zz_o[:,:,i])) *  deta_14_no_stress_dx_o[:,i] - self.alphaM4_o[i]/self.gammaM4_o[i]*F_14_no_stress_o[:,i] * (np.sinh(self.gammaM4_o[i]*(self.zz_o[:,:,i]+self.H_o[i])) + self.Av_o[i]*self.gammaM4_o[i]/self.Sf_o*np.cosh(self.gammaM4_o[i]*(self.zz_o[:,:,i]+self.H_o[i])))

    def adv_M4(self):
        coefM4_r = -1j * self.g / (2*self.omega) * (self.H_r - self.alphaM4_r / self.gammaM4_r * np.sinh(self.gammaM4_r*self.H_r)) 
        coefM4_m = -1j * self.g / (2*self.omega) * (self.H_m - self.alphaM4_m / self.gammaM4_m * np.sinh(self.gammaM4_m*self.H_m)) 
        coefM4_o = -1j * self.g / (2*self.omega) * (self.H_o - self.alphaM4_o / self.gammaM4_o * np.sinh(self.gammaM4_o*self.H_o)) 
        G1_r = np.exp(-self.gammaM4_r*self.H_r) + np.exp(self.gammaM4_r*self.H_r)+self.Av_r*self.gammaM4_r/self.Sf_r*(np.exp(self.gammaM4_r*self.H_r)-np.exp(-self.gammaM4_r*self.H_r))
        G1_m = np.exp(-self.gammaM4_m*self.H_m) + np.exp(self.gammaM4_m*self.H_m)+self.Av_m*self.gammaM4_m/self.Sf_m*(np.exp(self.gammaM4_m*self.H_m)-np.exp(-self.gammaM4_m*self.H_m))
        G1_o = np.exp(-self.gammaM4_o*self.H_o) + np.exp(self.gammaM4_o*self.H_o)+self.Av_o*self.gammaM4_o/self.Sf_o*(np.exp(self.gammaM4_o*self.H_o)-np.exp(-self.gammaM4_o*self.H_o))
        D1_r = -np.exp(self.gammaM4_r*self.H_r)*(1+self.Av_r*self.gammaM4_r/self.Sf_o)/G1_r
        D2_r = np.exp(-self.gammaM4_r*self.H_r)*(1-self.Av_r*self.gammaM4_r/self.Sf_o)/G1_r
        D1_m = -np.exp(self.gammaM4_m*self.H_m)*(1+self.Av_m*self.gammaM4_m/self.Sf_m)/G1_m
        D2_m = np.exp(-self.gammaM4_m*self.H_m)*(1-self.Av_m*self.gammaM4_m/self.Sf_m)/G1_m        
        D1_o = -np.exp(self.gammaM4_o*self.H_o)*(1+self.Av_o*self.gammaM4_o/self.Sf_o)/G1_o
        D2_o = np.exp(-self.gammaM4_o*self.H_o)*(1-self.Av_o*self.gammaM4_o/self.Sf_o)/G1_o
        
        W_r = -2 * self.gammaM4_r
        W_m = -2 * self.gammaM4_m
        W_o = -2 * self.gammaM4_o
        

        F_14_adv_r = np.zeros((len(self.z_r), len(self.x_r), len(self.H_r)), dtype = complex)
        F_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        uA_r = np.zeros((len(self.z_r), len(self.x_r), len(self.H_r)), dtype = complex)
        uB_r = np.zeros((len(self.z_r), len(self.x_r), len(self.H_r)), dtype = complex)
        uAx_r = np.zeros((len(self.z_r), len(self.x_r), len(self.H_r)), dtype = complex)
        uBx_r = np.zeros((len(self.z_r), len(self.x_r), len(self.H_r)), dtype = complex)
        G_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        u_adv_p_r = np.zeros((len(self.z_r), len(self.x_r), len(self.H_r)), dtype = complex)
        up_int_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        chi_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        dchidx_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        betaM4_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        CM4_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        for i in range(len(self.eta0_r[1])):
            F_14_adv_r[:,:,i] = 0.5 * (self.u0_r[:,:,i] * self.du0dx_r[:,:,i] + self.w0_r[:,:,i] * self.du0dz_r[:,:,i]) / self.Av_r[i]        
            uAx_r[:,:,i] = -F_14_adv_r[:,:,i] * np.exp(-self.gammaM4_r[i]*self.zz_r[:,:,i]) / W_r[i]
            uBx_r[:,:,i] = F_14_adv_r[:,:,i] * np.exp(self.gammaM4_r[i]*self.zz_r[:,:,i]) / W_r[i]
            uA_r[:,:,i] = integrate.cumtrapz(uAx_r[:,:,i], x=self.z_r[:,i], axis=0, initial=0)
            uB_r[:,:,i] = integrate.cumtrapz(uBx_r[:,:,i], x=self.z_r[:,i], axis=0, initial=0)
            G_r[:,i] = uA_r[-1,:,i] - uB_r[-1,:,i]
            u_adv_p_r[:,:,i] = uA_r[:,:,i] * np.exp(self.gammaM4_r[i]*self.zz_r[:,:,i]) + uB_r[:,:,i] * np.exp(-self.gammaM4_r[i]*self.zz_r[:,:,i])
            up_int_r[:,i] = np.trapz(u_adv_p_r[:,:,i], x=self.z_r[:,i], axis=0)
            chi_r[:,i] = -(up_int_r[:,i] - self.alphaM4_r[i] * G_r[:,i] * (
                (np.cosh(self.gammaM4_r[i]*self.H_r[i])-1)/self.gammaM4_r[i] + self.Av_r[i] / self.Sf_r * np.sinh(self.gammaM4_r[i]*self.H_r[i])))/ coefM4_r[i]
            dchidx_r[:,i] = np.gradient(chi_r[:,i], self.x_r[:,i])
            F_r[:,i] = (dchidx_r[:,i] + chi_r[:,i]/self.Lb_r[i]) 
            betaM4_r[:,i] =  np.trapz((1j*self.g/(2*self.omega)) * (np.exp(self.gammaM4_r[i] * self.zz_r[:,:,i])/G1_r[i] + np.exp(-self.gammaM4_r[i] * self.zz_r[:,:,i])/G1_r[i] - 1), x=self.z_r[:,i], axis = 0)
            CM4_r[:,i] = np.trapz((D1_r[i] * G_r[:,i] * np.exp(self.gammaM4_r[i] * self.zz_r[:,:,i]) + D2_r[i] * G_r[:,i] * np.exp(-self.gammaM4_r[i] * self.zz_r[:,:,i]) + u_adv_p_r[:,:,i]), x=self.z_r[:,i], axis = 0)
     
  
        F_14_adv_m = np.zeros((len(self.z_m), len(self.x_m), len(self.H_m)), dtype = complex)
        F_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        uA_m = np.zeros((len(self.z_m), len(self.x_m), len(self.H_m)), dtype = complex)
        uB_m = np.zeros((len(self.z_m), len(self.x_m), len(self.H_m)), dtype = complex)
        uAx_m = np.zeros((len(self.z_m), len(self.x_m), len(self.H_m)), dtype = complex)
        uBx_m = np.zeros((len(self.z_m), len(self.x_m), len(self.H_m)), dtype = complex)
        G_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        u_adv_p_m = np.zeros((len(self.z_m), len(self.x_m), len(self.H_m)), dtype = complex)
        up_int_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        chi_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        dchidx_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        betaM4_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        CM4_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        for i in range(len(self.eta0_m[1])):
            F_14_adv_m[:,:,i] = 0.5 * (self.u0_m[:,:,i] * self.du0dx_m[:,:,i] + self.w0_m[:,:,i] * self.du0dz_m[:,:,i]) / self.Av_m[i]        
            uAx_m[:,:,i] = -F_14_adv_m[:,:,i] * np.exp(-self.gammaM4_m[i]*self.zz_m[:,:,i]) / W_m[i]
            uBx_m[:,:,i] = F_14_adv_m[:,:,i] * np.exp(self.gammaM4_m[i]*self.zz_m[:,:,i]) / W_m[i]
            uA_m[:,:,i] = integrate.cumtrapz(uAx_m[:,:,i], x=self.z_m[:,i], axis=0, initial=0)
            uB_m[:,:,i] = integrate.cumtrapz(uBx_m[:,:,i], x=self.z_m[:,i], axis=0, initial=0)
            G_m[:,i] = uA_m[-1,:,i] - uB_m[-1,:,i]
            u_adv_p_m[:,:,i] = uA_m[:,:,i] * np.exp(self.gammaM4_m[i]*self.zz_m[:,:,i]) + uB_m[:,:,i] * np.exp(-self.gammaM4_m[i]*self.zz_m[:,:,i])
            up_int_m[:,i] = np.trapz(u_adv_p_m[:,:,i], x=self.z_m[:,i], axis=0)
            chi_m[:,i] = -(up_int_m[:,i] - self.alphaM4_m[i] * G_m[:,i] * (
                (np.cosh(self.gammaM4_m[i]*self.H_m[i])-1)/self.gammaM4_m[i] + self.Av_m[i] / self.Sf_m * np.sinh(self.gammaM4_m[i]*self.H_m[i])))/ coefM4_m[i]
            dchidx_m[:,i] = np.gradient(chi_m[:,i], self.x_m[:,i])
            F_m[:,i] = (dchidx_m[:,i] + chi_m[:,i]/self.Lb_m[i]) 
            betaM4_m[:,i] =  np.trapz((1j*self.g/(2*self.omega)) * (np.exp(self.gammaM4_m[i] * self.zz_m[:,:,i])/G1_m[i] + np.exp(-self.gammaM4_m[i] * self.zz_m[:,:,i])/G1_m[i] - 1), x=self.z_m[:,i], axis = 0)
            CM4_m[:,i] = np.trapz((D1_m[i] * G_m[:,i] * np.exp(self.gammaM4_m[i] * self.zz_m[:,:,i]) + D2_m[i] * G_m[:,i] * np.exp(-self.gammaM4_m[i] * self.zz_m[:,:,i]) + u_adv_p_m[:,:,i]), x=self.z_m[:,i], axis = 0)
     
        
        F_14_adv_o = np.zeros((len(self.z_o), len(self.x_o), len(self.H_o)), dtype = complex)
        F_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        uA_o = np.zeros((len(self.z_o), len(self.x_o), len(self.H_o)), dtype = complex)
        uB_o = np.zeros((len(self.z_o), len(self.x_o), len(self.H_o)), dtype = complex)
        uAx_o = np.zeros((len(self.z_o), len(self.x_o), len(self.H_o)), dtype = complex)
        uBx_o = np.zeros((len(self.z_o), len(self.x_o), len(self.H_o)), dtype = complex)
        G_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        u_adv_p_o = np.zeros((len(self.z_o), len(self.x_o), len(self.H_o)), dtype = complex)
        up_int_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        chi_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        dchidx_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        betaM4_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        CM4_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        for i in range(len(self.eta0_o[1])):
            F_14_adv_o[:,:,i] = 0.5 * (self.u0_o[:,:,i] * self.du0dx_o[:,:,i] + self.w0_o[:,:,i] * self.du0dz_o[:,:,i]) / self.Av_o[i]        
            uAx_o[:,:,i] = -F_14_adv_o[:,:,i] * np.exp(-self.gammaM4_o[i]*self.zz_o[:,:,i]) / W_o[i]
            uBx_o[:,:,i] = F_14_adv_o[:,:,i] * np.exp(self.gammaM4_o[i]*self.zz_o[:,:,i]) / W_o[i]
            uA_o[:,:,i] = integrate.cumtrapz(uAx_o[:,:,i], x=self.z_o[:,i], axis=0, initial=0)
            uB_o[:,:,i] = integrate.cumtrapz(uBx_o[:,:,i], x=self.z_o[:,i], axis=0, initial=0)
            G_o[:,i] = uA_o[-1,:,i] - uB_o[-1,:,i]
            u_adv_p_o[:,:,i] = uA_o[:,:,i] * np.exp(self.gammaM4_o[i]*self.zz_o[:,:,i]) + uB_o[:,:,i] * np.exp(-self.gammaM4_o[i]*self.zz_o[:,:,i])
            up_int_o[:,i] = np.trapz(u_adv_p_o[:,:,i], x=self.z_o[:,i], axis=0)
            chi_o[:,i] = -(up_int_o[:,i] - self.alphaM4_o[i] * G_o[:,i] * (
                (np.cosh(self.gammaM4_o[i]*self.H_o[i])-1)/self.gammaM4_o[i] + self.Av_o[i] / self.Sf_o * np.sinh(self.gammaM4_o[i]*self.H_o[i])))/ coefM4_o[i]
            dchidx_o[:,i] = np.gradient(chi_o[:,i], self.x_o[:,i])
            F_o[:,i] = (dchidx_o[:,i] + chi_o[:,i]/self.Lb_o[i]) 
            betaM4_o[:,i] =  np.trapz((1j*self.g/(2*self.omega)) * (np.exp(self.gammaM4_o[i] * self.zz_o[:,:,i])/G1_o[i] + np.exp(-self.gammaM4_o[i] * self.zz_o[:,:,i])/G1_o[i] - 1), x=self.z_o[:,i], axis = 0)
            CM4_o[:,i] = np.trapz((D1_o[i] * G_o[:,i] * np.exp(self.gammaM4_o[i] * self.zz_o[:,:,i]) + D2_o[i] * G_o[:,i] * np.exp(-self.gammaM4_o[i] * self.zz_o[:,:,i]) + u_adv_p_o[:,:,i]), x=self.z_o[:,i], axis = 0)


        self.matching(F_r, betaM4_r[0,:], CM4_r, "river")
        self.matching(F_m, betaM4_m[0,:], CM4_m, "middle")
        self.matching(F_o, betaM4_o[0,:], CM4_o, "ocean")
        self.vertex()
        self.eta14_adv_r, self.eta14_adv_m, self.eta14_adv_o = self.eta14() 
 
        deta_14_adv_dx_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
        u_adv_h_r = np.zeros((len(self.z_r),len(self.x_r), len(self.H_r)), dtype = complex)
        self.u14_adv_r = np.zeros((len(self.z_o),len(self.x_o), len(self.H_r)), dtype = complex)
        for i in range(len(self.eta0_r[1])):
            deta_14_adv_dx_r[:,i] = self.C1_r[i]*self.p1_r[i]*np.exp(self.p1_r[i]*self.x_r[:,i]) + self.C2_r[i]*self.p2_r[i]*np.exp(self.p2_r[i]*self.x_r[:,i]) + self.Ar[:,i]*self.p1_r[i]*np.exp(self.p1_r[i]*self.x_r[:,i]) + self.Br[:,i]*self.p2_r[i]*np.exp(self.p2_r[i]*self.x_r[:,i]) + self.Ax_r[:,i]*np.exp(self.p1_r[i]*self.x_r[:,i]) + self.Bx_r[:,i]*np.exp(self.p2_r[i]*self.x_r[:,i])
            u_adv_h_r[:,:,i] = (1j*self.g/(2*self.omega)*deta_14_adv_dx_r[:,i]) * (np.exp(self.gammaM4_r[i] * self.zz_r[:,:,i])/G1_r[i] + np.exp(-self.gammaM4_r[i] * self.zz_r[:,:,i])/G1_r[i] - 1) + D1_r[i] * G_r[:,i] * np.exp(self.gammaM4_r[i] * self.zz_r[:,:,i]) + D2_r[i] * G_r[:,i] * np.exp(-self.gammaM4_r[i] * self.zz_r[:,:,i])
            self.u14_adv_r[:,:,i] = u_adv_h_r[:,:,i] + u_adv_p_r[:,:,i]     


        deta_14_adv_dx_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
        u_adv_h_m = np.zeros((len(self.z_m),len(self.x_m), len(self.H_m)), dtype = complex)
        self.u14_adv_m = np.zeros((len(self.z_o),len(self.x_o), len(self.H_m)), dtype = complex)
        for i in range(len(self.eta0_m[1])):
            deta_14_adv_dx_m[:,i] = self.C1_m[i]*self.p1_m[i]*np.exp(self.p1_m[i]*self.x_m[:,i]) + self.C2_m[i]*self.p2_m[i]*np.exp(self.p2_m[i]*self.x_m[:,i]) + self.Am[:,i]*self.p1_m[i]*np.exp(self.p1_m[i]*self.x_m[:,i]) + self.Bm[:,i]*self.p2_m[i]*np.exp(self.p2_m[i]*self.x_m[:,i]) + self.Ax_m[:,i]*np.exp(self.p1_m[i]*self.x_m[:,i]) + self.Bx_m[:,i]*np.exp(self.p2_m[i]*self.x_m[:,i])
            u_adv_h_m[:,:,i] = (1j*self.g/(2*self.omega)*deta_14_adv_dx_m[:,i]) * (np.exp(self.gammaM4_m[i] * self.zz_m[:,:,i])/G1_m[i] + np.exp(-self.gammaM4_m[i] * self.zz_m[:,:,i])/G1_m[i] - 1) + D1_m[i] * G_m[:,i] * np.exp(self.gammaM4_m[i] * self.zz_m[:,:,i]) + D2_m[i] * G_m[:,i] * np.exp(-self.gammaM4_m[i] * self.zz_m[:,:,i])
            self.u14_adv_m[:,:,i] = u_adv_h_m[:,:,i] + u_adv_p_m[:,:,i]     


        deta_14_adv_dx_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
        u_adv_h_o = np.zeros((len(self.z_o),len(self.x_o), len(self.H_o)), dtype = complex)
        self.u14_adv_o = np.zeros((len(self.z_o),len(self.x_o), len(self.H_o)), dtype = complex)
        for i in range(len(self.eta0_o[1])):
            deta_14_adv_dx_o[:,i] = self.C1_o[i]*self.p1_o[i]*np.exp(self.p1_o[i]*self.x_o[:,i]) + self.C2_o[i]*self.p2_o[i]*np.exp(self.p2_o[i]*self.x_o[:,i]) + self.Ao[:,i]*self.p1_o[i]*np.exp(self.p1_o[i]*self.x_o[:,i]) + self.Bo[:,i]*self.p2_o[i]*np.exp(self.p2_o[i]*self.x_o[:,i]) + self.Ax_o[:,i]*np.exp(self.p1_o[i]*self.x_o[:,i]) + self.Bx_o[:,i]*np.exp(self.p2_o[i]*self.x_o[:,i])
            u_adv_h_o[:,:,i] = (1j*self.g/(2*self.omega)*deta_14_adv_dx_o[:,i]) * (np.exp(self.gammaM4_o[i] * self.zz_o[:,:,i])/G1_o[i] + np.exp(-self.gammaM4_o[i] * self.zz_o[:,:,i])/G1_o[i] - 1) + D1_o[i] * G_o[:,i] * np.exp(self.gammaM4_o[i] * self.zz_o[:,:,i]) + D2_o[i] * G_o[:,i] * np.exp(-self.gammaM4_o[i] * self.zz_o[:,:,i])
            self.u14_adv_o[:,:,i] = u_adv_h_o[:,:,i] + u_adv_p_o[:,:,i]     


    def matching(self, F, beta, CM4, channel):
        if channel == "river":
            Ax_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
            Bx_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
            A_tmp_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
            B_tmp_r = np.zeros((len(self.x_r), len(self.H_r)), dtype = complex)
            for i in range(len(self.H_r)):
                Ax_r[:,i] = -F[:,i]*np.exp(self.p2_r[i]*self.x_r[:,i])/self.W_r[:,i]
                Bx_r[:,i] = F[:,i]*np.exp(self.p1_r[i]*self.x_r[:,i])/self.W_r[:,i]
                A_tmp_r[:,i] = cumulative_trapezoid(Ax_r[:,i], x=self.x_r[:,i], initial=0)
                B_tmp_r[:,i] = cumulative_trapezoid(Bx_r[:,i], x=self.x_r[:,i], initial=0)
            
            Ar = A_tmp_r - A_tmp_r[-1] 
            Br = B_tmp_r - B_tmp_r[-1] 
            self.Ar = Ar
            self.Br = Br
            self.Ax_r = Ax_r
            self.Bx_r = Bx_r
        
            self.A_r = -self.p2_r/np.exp(self.p2_r*self.L_r)/(self.p1_r-self.p2_r*np.exp(self.p1_r*self.L_r)/np.exp(self.p2_r*self.L_r))
            self.B_r = -(Ar[0]*self.p1_r + Br[0]*self.p2_r + Ax_r[0] + Bx_r[0])/(self.p1_r-self.p2_r*np.exp(self.p1_r*self.L_r)/np.exp(self.p2_r*self.L_r))
            self.C_r = (1-self.A_r*np.exp(self.p1_r*self.L_r))/np.exp(self.p2_r*self.L_r)
            self.D_r = -self.B_r*np.exp(self.p1_r*self.L_r)/np.exp(self.p2_r*self.L_r)
            self.C1_river = beta*(self.p1_r*self.A_r*np.exp(self.p1_r*self.L_r) + self.p2_r*self.C_r*np.exp(self.p2_r*self.L_r))
            self.C2_river = beta*(self.p1_r*self.B_r*np.exp(self.p1_r*self.L_r) + self.p2_r*self.D_r*np.exp(self.p2_r*self.L_r) + Ax_r[-1]*np.exp(self.p1_r*self.L_r) + Bx_r[-1]*np.exp(self.p2_r*self.L_r)) + CM4[-1]
            
        if channel == "ocean":
            Ax_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
            Bx_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
            A_tmp_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
            B_tmp_o = np.zeros((len(self.x_o), len(self.H_o)), dtype = complex)
            for i in range(len(self.H_o)):
                Ax_o[:,i] = -F[:,i]*np.exp(self.p2_o[i]*self.x_o[:,i])/self.W_o[:,i]
                Bx_o[:,i] = F[:,i]*np.exp(self.p1_o[i]*self.x_o[:,i])/self.W_o[:,i]
                A_tmp_o[:,i] = cumulative_trapezoid(Ax_o[:,i], x=self.x_o[:,i], initial=0)
                B_tmp_o[:,i] = cumulative_trapezoid(Bx_o[:,i], x=self.x_o[:,i], initial=0)
            
            Ao = A_tmp_o - A_tmp_o[-1] 
            Bo = B_tmp_o - B_tmp_o[-1]
            self.Ao = Ao
            self.Bo = Bo
            self.Ax_o = Ax_o
            self.Bx_o = Bx_o
            
            O = self.x_o[0]
            self.A_o = 1/(np.exp(self.p1_o*O)-np.exp(self.p1_o*self.L)/np.exp(self.p2_o*self.L)*np.exp(self.p2_o*O))
            self.B_o = -(Ao[0]*np.exp(self.p1_o*O) + Bo[0]*np.exp(self.p2_o*O))/(np.exp(self.p1_o*O)-np.exp(self.p1_o*self.L)/np.exp(self.p2_o*self.L)*np.exp(self.p2_o*O))
            self.C_o = -self.A_o*np.exp(self.p1_o*self.L)/np.exp(self.p2_o*self.L)
            self.D_o = -self.B_o*np.exp(self.p1_o*self.L)/np.exp(self.p2_o*self.L)
            self.C1_ocean = beta*(self.p1_o*self.A_o*np.exp(self.p1_o*O) + self.p2_o*self.C_o*np.exp(self.p2_o*O))
            self.C2_ocean = beta*(self.p1_o*self.B_o*np.exp(self.p1_o*O) + self.p2_o*self.D_o*np.exp(self.p2_o*O) + Ao[0]*self.p1_o*np.exp(self.p1_o*O) + Bo[0]*self.p2_o*np.exp(self.p2_o*O) + Ax_o[0]*np.exp(self.p1_o*O) + Bx_o[0]*np.exp(self.p2_o*O)) + CM4[-1]
        
        if channel == "middle": 
            Ax_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
            Bx_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
            A_tmp_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
            B_tmp_m = np.zeros((len(self.x_m), len(self.H_m)), dtype = complex)
            for i in range(len(self.H_m)):
                Ax_m[:,i] = -F[:,i]*np.exp(self.p2_m[i]*self.x_m[:,i])/self.W_m[:,i]
                Bx_m[:,i] = F[:,i]*np.exp(self.p1_m[i]*self.x_m[:,i])/self.W_m[:,i]
                A_tmp_m[:,i] = cumulative_trapezoid(Ax_m[:,i], x=self.x_m[:,i], initial=0)
                B_tmp_m[:,i] = cumulative_trapezoid(Bx_m[:,i], x=self.x_m[:,i], initial=0)
            
            Am = A_tmp_m - A_tmp_m[-1] 
            Bm = B_tmp_m - B_tmp_m[-1]
            self.Am = Am
            self.Bm = Bm
            self.Ax_m = Ax_m
            self.Bx_m = Bx_m
            

            M = self.x_m[-1]
            R = self.x_m[0]
            self.A_m = 1/(np.exp(self.p1_m*R)-np.exp(self.p2_m*R)/np.exp(self.p2_m*M)*np.exp(self.p1_m*M))
            self.B_m = -np.exp(self.p2_m*R)/np.exp(self.p2_m*M)/(np.exp(self.p1_m*R)-np.exp(self.p2_m*R)/np.exp(self.p2_m*M)*np.exp(self.p1_m*M))
            self.C_m = -(Am[0]*np.exp(self.p1_m*R) + Bm[0]*np.exp(self.p2_m*R))/(np.exp(self.p1_m*R)-np.exp(self.p2_m*R)/np.exp(self.p2_m*M)*np.exp(self.p1_m*M))
            self.D_m = -self.A_m*np.exp(self.p1_m*M)/np.exp(self.p2_m*M)
            self.E_m = (1-self.B_m*np.exp(self.p1_m*M))/np.exp(self.p2_m*M)
            self.F_m = -self.C_m*np.exp(self.p1_m*M)/np.exp(self.p2_m*M)
            
            self.C1_middle = beta*(self.A_m*self.p1_m * np.exp(self.p1_m*R) + self.D_m*self.p2_m * np.exp(self.p2_m*R))
            self.D1_middle =  beta*(self.B_m*self.p1_m * np.exp(self.p1_m*R) + self.E_m*self.p2_m * np.exp(self.p2_m*R))
            self.E1_middle =  beta*(self.C_m*self.p1_m * np.exp(self.p1_m*R) + self.F_m*self.p2_m * np.exp(self.p2_m*R) + Am[0]*self.p1_m*np.exp(self.p1_m*R) + Bm[0]*self.p2_m*np.exp(self.p2_m*R) + Ax_m[0]*np.exp(self.p1_m*R) + Bx_m[0]*np.exp(self.p2_m*R)) + CM4[0]
        
            self.C2_middle = beta*(self.A_m*self.p1_m * np.exp(self.p1_m*M) + self.D_m*self.p2_m * np.exp(self.p2_m*M))
            self.D2_middle =  beta*(self.B_m*self.p1_m * np.exp(self.p1_m*M) + self.E_m*self.p2_m * np.exp(self.p2_m*M))
            self.E2_middle =  beta*(self.C_m*self.p1_m * np.exp(self.p1_m*M) + self.F_m*self.p2_m * np.exp(self.p2_m*M) + Ax_m[-1]*np.exp(self.p1_m*M) + Bx_m[-1]*np.exp(self.p2_m*M)) + CM4[-1]

    def vertex(self):
        C_v1 = np.sum(self.B_rx[-1,0] * self.C1_river[0]) - np.sum(self.B_mx[0,0] * self.C1_middle[0]) - np.sum(self.B_mx[0,1] * self.C1_middle[1])
        D_v1 = -np.sum(self.B_mx[0,1] * self.D1_middle[1])
        E_v1 = -np.sum(self.B_mx[0,0] * self.D1_middle[0])
        F_v1 = np.sum(self.B_mx[0,1] * self.E1_middle[1]) + np.sum(self.B_mx[0,0] * self.E1_middle[0]) - np.sum(self.B_rx[-1,0] * self.C2_river[0])
        C_v2 = np.sum(self.B_mx[-1,1] * self.C2_middle[1]) 
        D_v2 = np.sum(self.B_rx[-1,1] * self.C1_river[1]) + np.sum(self.B_mx[-1,1] * self.D2_middle[1]) - np.sum(self.B_mx[0,2] * self.C1_middle[2])
        E_v2 = -np.sum(self.B_mx[0,2] * self.D1_middle[2])
        F_v2 = -np.sum(self.B_rx[-1,1] * self.C2_river[1]) - np.sum(self.B_mx[-1,1] * self.E2_middle[1]) + np.sum(self.B_mx[0,2] * self.E1_middle[2])
        C_v3 = np.sum(self.B_mx[-1,0] * self.C2_middle[0]) 
        D_v3 = np.sum(self.B_mx[-1,2] * self.C2_middle[2])
        E_v3 = -np.sum(self.B_ox[0,:]*self.C1_ocean) + np.sum(self.B_mx[-1,0] * self.D2_middle[0]) + np.sum(self.B_mx[-1,2] * self.D2_middle[2])
        F_v3 = np.sum(self.B_ox[0,:]*self.C2_ocean) - np.sum(self.B_mx[-1,0] * self.E2_middle[0]) - np.sum(self.B_mx[-1,2] * self.E2_middle[2])
        
        a = np.array([[C_v1, D_v1, E_v1], [C_v2, D_v2, E_v2], [C_v3, D_v3, E_v3]])
        b = np.array([F_v1, F_v2, F_v3])
        
        self.eta_vertex_1 = np.linalg.solve(a,b)[0]
        self.eta_vertex_2 = np.linalg.solve(a,b)[1]
        self.eta_vertex_3 = np.linalg.solve(a,b)[2]
        
    def eta14(self):
        self.C1_r1 = self.eta_vertex_1 * self.A_r[0] + self.B_r[0]
        self.C2_r1 = self.eta_vertex_1 * self.C_r[0] + self.D_r[0]
        self.C1_r2 = self.eta_vertex_2 * self.A_r[1] + self.B_r[1]
        self.C2_r2 = self.eta_vertex_2 * self.C_r[1] + self.D_r[1]
        self.C1_r = np.column_stack((self.C1_r1, self.C1_r2))[0]
        self.C2_r = np.column_stack((self.C2_r1, self.C2_r2))[0]
        
        self.C1_o1 = self.eta_vertex_3 * self.A_o[0] + self.B_o[0]  
        self.C2_o1 = self.eta_vertex_3 * self.C_o[0] + self.D_o[0]
        self.C1_o2 = self.eta_vertex_3 * self.A_o[1] + self.B_o[1]  
        self.C2_o2 = self.eta_vertex_3 * self.C_o[1] + self.D_o[1]
        self.C1_o = np.column_stack((self.C1_o1, self.C1_o2))[0]
        self.C2_o = np.column_stack((self.C2_o1, self.C2_o2))[0]
        
        self.C1_m1 = self.eta_vertex_1* self.A_m[0] + self.eta_vertex_3 * self.B_m[0] + self.C_m[0]
        self.C2_m1 = self.eta_vertex_1* self.D_m[0] + self.eta_vertex_3 * self.E_m[0] + self.F_m[0]
        self.C1_m2 = self.eta_vertex_1* self.A_m[1] + self.eta_vertex_2 * self.B_m[1] + self.C_m[1]
        self.C2_m2 = self.eta_vertex_1* self.D_m[1] + self.eta_vertex_2 * self.E_m[1] + self.F_m[1]   
        self.C1_m3 = self.eta_vertex_2* self.A_m[2] + self.eta_vertex_3 * self.B_m[2] + self.C_m[2]
        self.C2_m3 = self.eta_vertex_2* self.D_m[2] + self.eta_vertex_3 * self.E_m[2] + self.F_m[2]
        self.C1_m = np.column_stack((self.C1_m1, self.C1_m2, self.C1_m3))[0]
        self.C2_m = np.column_stack((self.C2_m1, self.C2_m2, self.C2_m3))[0]
        
        eta_h_r = self.C1_r*np.exp(self.p1_r*self.x_r) + self.C2_r*np.exp(self.p2_r*self.x_r)
        eta_p_r = self.Ar * np.exp(self.p1_r*self.x_r) + self.Br * np.exp(self.p2_r*self.x_r)
        eta_14_r = eta_h_r + eta_p_r
        
        eta_h_m = self.C1_m*np.exp(self.p1_m*self.x_m) + self.C2_m*np.exp(self.p2_m*self.x_m)
        eta_p_m = self.Am * np.exp(self.p1_m*self.x_m) + self.Bm * np.exp(self.p2_m*self.x_m)
        eta_14_m = eta_h_m + eta_p_m
        
        eta_h_o = self.C1_o*np.exp(self.p1_o*self.x_o) + self.C2_o*np.exp(self.p2_o*self.x_o)
        eta_p_o = self.Ao * np.exp(self.p1_o*self.x_o) + self.Bo * np.exp(self.p2_o*self.x_o)
        eta_14_o = eta_h_o + eta_p_o
        return eta_14_r, eta_14_m, eta_14_o