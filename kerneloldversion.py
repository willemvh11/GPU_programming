#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:10:19 2018

@author: willemvh
"""

def kernelprecessnucspins(dt):
    return """
    static void rot( float *w, __local float *vec)
    {
        float mw = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
        float dt = """+str(dt)+""";
        float omega[3];
        float invmw = 1.0f/mw;
        omega[0] = w[0]*invmw;
        omega[1] = w[1]*invmw;
        omega[2] = w[2]*invmw;
        float dot = omega[0]*vec[0] + omega[1]*vec[1] + omega[2]*vec[2];
        float i1[3];
        i1[0] = omega[0]*dot;
        i1[1] = omega[1]*dot;
        i1[2] = omega[2]*dot;
        float i2[3];
        i2[0] = vec[0] - i1[0];
        i2[1] = vec[1] - i1[1];
        i2[2] = vec[2] - i1[2];
        float i3[3];
        i3[0] = omega[1]*vec[2] - omega[2]*vec[1];
        i3[1] = omega[2]*vec[0] - omega[0]*vec[2];
        i3[2] = omega[0]*vec[1] - omega[1]*vec[0];
        float cwt =cos(mw*dt);
        float swt =sin(mw*dt);
        vec[0] = i1[0] + i2[0]*cwt + i3[0]*swt;
        vec[1] = i1[1] + i2[1]*cwt + i3[1]*swt;
        vec[2] = i1[2] + i2[2]*cwt + i3[2]*swt;
    }
    
    __kernel void precessnucspins(__global float *i, __global float *a,
    __global float *s, const int ni, __local float *iloc)
    {
        int ggid = get_global_id(0);
        int ggid1 = get_global_id(1);
        int glid = get_local_id(0);
        int glid1 = get_local_id(1);
        int groupid = get_group_id(0);
        int nl = get_local_size(0);
        float w[3];
        float itemp[3];  
        float store = a[ggid];
        w[0] = store*s[3*ggid1];
        w[1] = store*s[1 + 3*ggid1];
        w[2] = store*s[2 + 3*ggid1];
        int sind = 3*nl*groupid + glid;
        for(int ii = 0; ii < 3 && sind < 3*ni; ++ii, sind += nl)
        {
            iloc[glid + ii*nl + 3*nl*glid1] = i[sind + 3*ni*ggid1];
        }
        rot (w, iloc+(3*glid+3*nl*glid1));
        iloc[3*glid + 3*nl*glid1] = itemp[0];
        iloc[3*glid + 1 + 3*nl*glid1] = itemp[1];
        iloc[3*glid + 2 + 3*nl*glid1] = itemp[2];
        int wind = 3*nl*groupid + glid;
        for(int ii = 0; ii < 3 && wind < 3*ni; ++ii, wind += nl)
        {
            i[wind + 3*ni*ggid1] = iloc[glid + 3*nl*glid1 + ii*nl];
        }
    }
    """
    
def kernelprecesselecspins(dt):
    return """
        static void rot(float *w, __local float *vec)
    {
        float mw; 
        mw = sqrt(pown(w[0],2) + pown(w[1],2) + pown(w[2],2));
        float dt;
        float invmw = 1.0f/mw;
        dt = """+str(dt)+""";
        float omega[3];
        omega[0] = w[0]*invmw;
        omega[1] = w[1]*invmw;
        omega[2] = w[2]*invmw;
        float dot;
        dot = omega[0]*vec[0] + omega[1]*vec[1] + omega[2]*vec[2];
        float i1[3];
        i1[0] = omega[0]*dot;
        i1[1] = omega[1]*dot;
        i1[2] = omega[2]*dot;
        float i2[3];
        i2[0] = vec[0] - i1[0];
        i2[1] = vec[1] - i1[1];
        i2[2] = vec[2] - i1[2];
        float i3[3];
        i3[0] = omega[1]*vec[2] - omega[2]*vec[1];
        i3[1] = omega[2]*vec[0] - omega[0]*vec[2];
        i3[2] = omega[0]*vec[1] - omega[1]*vec[0];
        float cwt =cos(mw*dt);
        float swt =sin(mw*dt);
        vec[0] = i1[0] + i2[0]*cwt + i3[0]*swt;
        vec[1] = i1[1] + i2[1]*cwt + i3[1]*swt;
        vec[2] = i1[2] + i2[2]*cwt + i3[2]*swt;
    }
    
    __kernel void precesselecspins(__global float *w, __global float *wi, 
    __global float *s, const int size, const int x, __global float *sstore,
    __local float *sloc, __local float *wloc, const int a)
    {
        float wtemp[3];
        int ggid = get_global_id(0);
        int glid = get_local_id(0);
        int nl = get_local_size(0);
        int groupid = get_group_id(0);
        int sind = 3*nl*groupid + glid;
        for(int ii = 0; ii < 3; ++ii, sind += nl)
        {
            sloc[glid + ii*nl] = s[sind];
        }
        wloc[3*glid] = w[3*a*ggid];
        wloc[3*glid + 1] = w[3*a*ggid + 1];
        wloc[3*glid + 2] = w[3*a*ggid + 2];
        sstore[3*x + 3*size*ggid] = sloc[3*glid];
        sstore[3*x + 1 + 3*size*ggid] = sloc[3*glid + 1];
        sstore[3*x + 2 + 3*size*ggid] = sloc[3*glid + 2];
        wtemp[0] = wloc[3*glid] + wi[0];
        wtemp[1] = wloc[1 + 3*glid] + wi[1];
        wtemp[2] = wloc[2 + 3*glid] + wi[2];
        rot (wtemp, sloc+(3*glid));
        int wind = 3*nl*groupid + glid;
        for(int ii = 0; ii < 3; ++ii, wind += nl)
        {
            s[wind] = sloc[glid + ii*nl];
        }
    }
    """
    
def kernelreduce():
    return """
    __kernel void reduce(__global float *i, __global float *w, const int n, const int a, __local float *wtemp,
    __global float *hyp, const int ni, __local float *store)
    {
        int ggid = get_global_id(0);
        int ggid1 = get_global_id(1);
        int glid = get_local_id(0);
        int glid1 = get_local_id(1);
        int nl = get_local_size(0);
        int ng = get_global_size(0);
        int groupid = get_group_id(0);
        store[glid] = 1;
        if (a == ng) 
        {
            store[glid] = hyp[ggid];
            int sind = 3*nl*groupid + glid;
            for(int ii = 0; ii < 3 && sind < 3*ni; ++ii, sind += nl)
            {
                w[sind + 3*ng*ggid1] = i[sind + 3*ni*ggid1];
            }
        }
        wtemp[glid + 3*nl*glid1] = store[(glid - glid%3)/3]*w[3*nl*groupid + glid + 3*ng*ggid1];
        wtemp[glid + nl + 3*nl*glid1] = store[(glid + nl - (glid + nl)%3)/3]*w[3*nl*groupid + glid + nl + 3*ng*ggid1];
        wtemp[glid + 2*nl + 3*nl*glid1] = store[(glid + 2*nl - (glid + 2*nl)%3)/3]*w[3*nl*groupid + glid + 2*nl + 3*ng*ggid1];
        #pragma unroll
        for (int k=1; k < n; k++)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            int b = nl >> k;
            if (glid < b) 
            {
                wtemp[3*glid + 3*nl*glid1] += wtemp[3*(glid + b)+ 3*nl*glid1];
                wtemp[3*glid + 1 + 3*nl*glid1] += wtemp[3*(glid + b) + 1+ 3*nl*glid1];
                wtemp[3*glid + 2 + 3*nl*glid1] += wtemp[3*(glid + b) + 2+ 3*nl*glid1];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (glid == 0) 
        {
            w[(ggid >> n)*3+ 3*ng*ggid1] = wtemp[3*nl*glid1] + wtemp[3 + 3*nl*glid1];
            w[(ggid >> n)*3 + 1 + 3*ng*ggid1] = wtemp[1 + 3*nl*glid1] + wtemp[4 + 3*nl*glid1];
            w[(ggid >> n)*3 + 2 + 3*ng*ggid1] = wtemp[2 + 3*nl*glid1] + wtemp[5 + 3*nl*glid1];
        }
        int e = a%nl;
        int c = (a - e)/nl + e;
        int wind = 3*nl*groupid + glid + 3*c;
        for (int ii = 0; ii < 3 && wind < 3*ng; ++ii, wind += nl)
        {
            w[wind + 3*ng*ggid1] = 0;
        }
    }
    """

def kernelvecbuilds():
    return """
    __kernel void vecbuilds(__global float *s, __global float *v, const int ni,
    __local float *sloc, __global float *sinit)
    {
        int ggid = get_global_id(0);
        int nl = get_local_size(0);
        int glid = get_local_id(0);
        int groupid = get_group_id(0);
        float m = sqrt(3.0f/4.0f);
        float phi = 2.0*M_PI_F*v[2*(ni+1)*ggid];
        float th = acos(2.0f*v[1 + 2*(ni+1)*ggid] - 1.0f);
        sloc[3*glid] = m*sin(th)*cos(phi);
        sloc[3*glid + 1] = m*sin(th)*sin(phi);
        sloc[3*glid + 2] = m*cos(th);
        int wind = 3*nl*groupid + glid;
        for(int ii = 0; ii < 3; ++ii, wind += nl)
        {
            s[wind] = sloc[glid + ii*nl];
            sinit[wind] = sloc[glid + ii*nl];
        }
    }
    """
    
def kernelvecbuildi():
    return """
    __kernel void vecbuildi(__global float *i, __global float *v, const int ni, 
    __local float *vloc, __local float *iloc)
    {
        int ggid1 = get_global_id(1);
        int nl = get_local_size(0);
        int glid = get_local_id(0);
        int glid1 = get_local_id(1);
        int groupid = get_group_id(0);
        float m = sqrt(3.0f/4.0f);
        int sind = 2*nl*groupid + glid;
        for (int ii = 0; ii < 2 && sind < 2*ni; ++ii, sind += nl)
        {
            vloc[glid + ii*nl + 2*nl*glid1] = v[sind + 2*(ni+1)*ggid1 + 2];
        }
        float phi = 2.0f*M_PI_F*vloc[2*glid + 2*nl*glid1];
        float th = acos(2.0f*vloc[2*glid + 2*nl*glid1 + 1] - 1.0f);
        iloc[3*glid + 3*nl*glid1] = m*sin(th)*cos(phi);
        iloc[3*glid + 3*nl*glid1 + 1] = m*sin(th)*sin(phi);
        iloc[3*glid + 3*nl*glid1 + 2] = m*cos(th);
        int wind = 3*nl*groupid + glid;
        for(int ii = 0; ii < 3 && wind < 3*ni; ++ii, wind += nl)
        {
            i[wind + 3*ni*ggid1] = iloc[glid + 3*nl*glid1 + ii*nl];
        }
        
    }
    
    """
    
def kerneltensors():
    return """
    __kernel void tensors(__global float *output, __global float *Rxx,
    __global float *Rxy, __global float *Rzz, const int size, const int j)
    {
         int ggid = get_global_id(0);
         if (ggid < size) 
         {
            Rxx[ggid + j*size] = Rxx[ggid + j*size] + output[3*ggid];
            Rxy[ggid + j*size] = Rxy[ggid + j*size] + output[3*ggid + 1];
            Rzz[ggid + j*size] = Rzz[ggid + j*size] + output[3*ggid + 2]; 
         }
    }
    """

def kernelprep2():
    return """
    __kernel void prep2(__global float *sstore, __global float *output, const int size,
    __local float *sstoreloc, __local float *outputloc, __global float *sinit)
    {
        int ggid1 = get_global_id(1);
        int ng = get_global_size(0);
        int glid = get_local_id(0);
        int glid1 = get_local_id(1);
        int nl = get_local_size(0);
        int groupid = get_group_id(0);
        float store[2];
        store[0] = sinit[3*ggid1];
        store[1] = sinit[3*ggid1 + 2];
        int sind = 3*nl*groupid + glid;
        for (int ii = 0; ii < 3 && sind < 3*size; ++ii, sind += nl)
        {
            sstoreloc[glid + ii*nl + 3*nl*glid1] = sstore[sind + 3*size*ggid1];
        }
        outputloc[glid*3 + 3*nl*glid1] = store[0]*sstoreloc[glid*3 + 3*nl*glid1];
        outputloc[glid*3 + 3*nl*glid1 + 1] = store[0]*sstoreloc[glid*3 + 3*nl*glid1 + 1];
        outputloc[glid*3 + 3*nl*glid1 + 2] = store[1]*sstoreloc[glid*3 + 3*nl*glid1 + 2];
        int wind = 3*nl*groupid + glid;
        for(int ii = 0; ii < 3 && wind < 3*size; ++ii, wind += nl)
        {
            output[wind + 3*ng*ggid1] = outputloc[glid + 3*nl*glid1 + ii*nl];
        }
        
    }
    """
    
def kernelreduce2():
    return """
    __kernel void reduce2(const int n, const int a, __local float *sstoretemp, 
    __global float *output)
    {
        int ggid1 = get_global_id(1);
        int glid = get_local_id(0);
        int glid1 = get_local_id(1);
        int nl1 = get_local_size(1);
        int nl = get_local_size(0);
        int ng = get_global_size(0);
        int groupid = get_group_id(0);
        int sind = 3*nl*groupid + glid;
        sstoretemp[glid + 3*nl*glid1] = output[sind + 3*ng*ggid1];
        sstoretemp[glid + 3*nl*glid1 + nl] = output[sind + 3*ng*ggid1 + nl];
        sstoretemp[glid + 3*nl*glid1 + 2*nl] = output[sind + 3*ng*ggid1 + 2*nl];
        #pragma unroll
        for (int k=1; k < n; k++)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            int b = nl1 >> k;
            if (glid1 < b) 
            {
                sstoretemp[3*glid + 3*nl*glid1] += sstoretemp[3*glid+ 3*nl*(glid1+b)];
                sstoretemp[3*glid + 1 + 3*nl*glid1] += sstoretemp[3*glid + 1 + 3*nl*(glid1+b)];
                sstoretemp[3*glid + 2 + 3*nl*glid1] += sstoretemp[3*glid + 2 + 3*nl*(glid1+b)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (glid1 == 0) 
        {
            output[sind + 3*ng*(ggid1 >> n)] = sstoretemp[glid] + sstoretemp[glid + 3*nl];
            output[sind + nl + 3*ng*(ggid1 >> n)] = sstoretemp[glid + nl] + sstoretemp[glid + nl + 3*nl];
            output[sind + 2*nl + 3*ng*(ggid1 >> n)] = sstoretemp[glid + 2*nl] + sstoretemp[glid + 2*nl + 3*nl];
        }
        int e = a%nl1;
        int c = (a - e)/nl1 + e - 1;
        if (ggid1 > c) 
        {
            for (int ii = 0; ii < 3 && sind < 3*ng; ++ii, sind += nl)
            {
                output[sind + 3*ng*ggid1] = 0;
            }
        }
    }
    """
    
def kernelrandnum():
    return """
    __kernel void randnum(__global float *v)
    {
        int ggid = get_global_id(0);
        uint randoms[2];
        randoms[0] = 1;
        randoms[1] = 2;
        uint seed = randoms.x + ggid;
        uint t = seed ^ (seed << 11);  
        uint result = randoms.y ^ (randoms.y >> 19) ^ (t ^ (t >> 8));
        v[ggid] = result;
    }
    """