#include<gputi/root_finder.h>

// calculate a*(2^b)
__device__ long power(const long a, const int b) { return a << b; }
__device__ Scalar Numccd2double(const Numccd &n)
{
    Scalar r = Scalar(n.first) / power(1, n.second);
    return r;
}
__device__ VectorMax3d width(const Interval3 &x)
{
    VectorMax3d w;
    for (int i = 0; i < 3; i++)
    {
        w.v[i] =// 0.1;
        (Numccd2double(x[i].second) - Numccd2double(x[i].first));
    }
    return w;
}

__device__ void convert_tuv_to_array(
        const Interval3 &itv,
        std::array<Scalar, 8> &t_up,
        std::array<Scalar, 8> &t_dw,
        std::array<Scalar, 8> &u_up,
        std::array<Scalar, 8> &u_dw,
        std::array<Scalar, 8> &v_up,
        std::array<Scalar, 8> &v_dw)
    {
        // t order: 0,0,0,0,1,1,1,1
        // u order: 0,0,1,1,0,0,1,1
        // v order: 0,1,0,1,0,1,0,1
        Scalar t0_up = itv[0].first.first,
               t0_dw = power(1, itv[0].first.second),
               t1_up = itv[0].second.first,
               t1_dw = power(1, itv[0].second.second),

               u0_up = itv[1].first.first,
               u0_dw = power(1, itv[1].first.second),
               u1_up = itv[1].second.first,
               u1_dw = power(1, itv[1].second.second),

               v0_up = itv[2].first.first,
               v0_dw = power(1, itv[2].first.second),
               v1_up = itv[2].second.first,
               v1_dw = power(1, itv[2].second.second);
        t_up = {{t0_up, t0_up, t0_up, t0_up, t1_up, t1_up, t1_up, t1_up}};
        t_dw = {{t0_dw, t0_dw, t0_dw, t0_dw, t1_dw, t1_dw, t1_dw, t1_dw}};
        u_up = {{u0_up, u0_up, u1_up, u1_up, u0_up, u0_up, u1_up, u1_up}};
        u_dw = {{u0_dw, u0_dw, u1_dw, u1_dw, u0_dw, u0_dw, u1_dw, u1_dw}};
        v_up = {{v0_up, v1_up, v0_up, v1_up, v0_up, v1_up, v0_up, v1_up}};
        v_dw = {{v0_dw, v1_dw, v0_dw, v1_dw, v0_dw, v1_dw, v0_dw, v1_dw}};
    }

__device__ Scalar* function_vf(
        const Scalar &vs,
        const Scalar &t0s,
        const Scalar &t1s,
        const Scalar &t2s,
        const Scalar &ve,
        const Scalar &t0e,
        const Scalar &t1e,
        const Scalar &t2e,
        const Scalar *t_up,
        const Scalar *t_dw,
        const Scalar *u_up,
        const Scalar *u_dw,
        const Scalar *v_up,
        const Scalar *v_dw)
    {
        Scalar *rst=new Scalar[8];
        for (int i = 0; i < 8; i++)
        {
            Scalar v = (ve - vs) * t_up[i] / t_dw[i] + vs;
            Scalar t0 = (t0e - t0s) * t_up[i] / t_dw[i] + t0s;
            Scalar t1 = (t1e - t1s) * t_up[i] / t_dw[i] + t1s;
            Scalar t2 = (t2e - t2s) * t_up[i] / t_dw[i] + t2s;
            Scalar pt = (t1 - t0) * u_up[i] / u_dw[i]
                        + (t2 - t0) * v_up[i] / v_dw[i] + t0;
            rst[i] = v - pt;
        }
        return rst;
    }