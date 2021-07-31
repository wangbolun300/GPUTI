#include<gputi/root_finder.h>

// calculate a*(2^b)
__device__ Singleinterval::Singleinterval(Numccd f, Numccd s){
    first=f;
    second=s;
}
__device__ long power(const long a, const int b) { return a << b; }
__device__ Scalar Numccd2double(const Numccd &n)
{
    Scalar r = Scalar(n.first) / power(1, n.second);
    return r;
}
__device__ VectorMax3d width(const Singleinterval* x)
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
        const Singleinterval* itv0, Singleinterval* itv1, Singleinterval* itv2, 
        Scalar* t_up,
        Scalar* t_dw,
        Scalar* u_up,
        Scalar* u_dw,
        Scalar* v_up,
        Scalar* v_dw)
    {
        // t order: 0,0,0,0,1,1,1,1
        // u order: 0,0,1,1,0,0,1,1
        // v order: 0,1,0,1,0,1,0,1
        Scalar t0_up = itv0->first.first,
               t0_dw = power(1, itv0->first.second),
               t1_up = itv0->second.first,
               t1_dw = power(1, itv0->second.second),

               u0_up = itv1->first.first,
               u0_dw = power(1, itv1->first.second),
               u1_up = itv1->second.first,
               u1_dw = power(1, itv1->second.second),

               v0_up = itv2->first.first,
               v0_dw = power(1, itv2->first.second),
               v1_up = itv2->second.first,
               v1_dw = power(1, itv2->second.second);
        
        t_up[0]=t0_up;t_up[1]= t0_up;t_up[2]= t0_up;t_up[3]= t0_up;t_up[4]= t1_up; t_up[5]=t1_up;t_up[6]= t1_up;t_up[7]= t1_up;
        t_dw[0]=t0_dw;t_dw[1]= t0_dw;t_dw[2]= t0_dw;t_dw[3]= t0_dw;t_dw[4]= t1_dw; t_dw[5]=t1_dw;t_dw[6]= t1_dw;t_dw[7]= t1_dw;
        u_up[0]=u0_up;u_up[1]= u0_up;u_up[2]= u1_up;u_up[3]= u1_up;u_up[4]= u0_up; u_up[5]=u0_up;u_up[6]= u1_up;u_up[7]= u1_up;
        u_dw[0]=u0_dw;u_dw[1]= u0_dw;u_dw[2]= u1_dw;u_dw[3]= u1_dw;u_dw[4]= u0_dw; u_dw[5]=u0_dw;u_dw[6]= u1_dw;u_dw[7]= u1_dw;
        v_up[0]=v0_up;v_up[1]= v1_up;v_up[2]= v0_up;v_up[3]= v1_up;v_up[4]= v0_up; v_up[5]=v1_up;v_up[6]= v0_up;v_up[7]= v1_up;
        v_dw[0]=v0_dw;v_dw[1]= v1_dw;v_dw[2]= v0_dw;v_dw[3]= v1_dw;v_dw[4]= v0_dw; v_dw[5]=v1_dw;v_dw[6]= v0_dw;v_dw[7]= v1_dw;

    }
// __device__ void test_convert_tuv_to_array(
//         const Singleinterval* itv,
//         Scalar* t_up,
//         Scalar* t_dw,
//         Scalar* u_up,
//         Scalar* u_dw,
//         Scalar* v_up,
//         Scalar* v_dw){
//         std::array<Scalar, 8> t0,t1,u0,u1,v0,v1;
//         convert_tuv_to_array(itv,t0,t1,u0,u1,v0,v1);
//         for(int i=0;i<8;i++){
//             t_up[i]=t0[i];
//             t_dw[i]=t1[i];
//             u_up[i]=u0[i];
//             u_dw[i]=u1[i];
//             v_up[i]=v0[i];
//             v_dw[i]=v1[i];
//         }
// }
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

__device__ Scalar* function_ee(
        const Scalar &a0s,
        const Scalar &a1s,
        const Scalar &b0s,
        const Scalar &b1s,
        const Scalar &a0e,
        const Scalar &a1e,
        const Scalar &b0e,
        const Scalar &b1e,
        const Scalar* t_up,
        const Scalar* t_dw,
        const Scalar* u_up,
        const Scalar* u_dw,
        const Scalar* v_up,
        const Scalar* v_dw)
    {
        Scalar* rst=new Scalar[8];
        for (int i = 0; i < 8; i++)
        {
            Scalar edge0_vertex0 = (a0e - a0s) * t_up[i] / t_dw[i] + a0s;
            Scalar edge0_vertex1 = (a1e - a1s) * t_up[i] / t_dw[i] + a1s;
            Scalar edge1_vertex0 = (b0e - b0s) * t_up[i] / t_dw[i] + b0s;
            Scalar edge1_vertex1 = (b1e - b1s) * t_up[i] / t_dw[i] + b1s;

            Scalar edge0_vertex =
                (edge0_vertex1 - edge0_vertex0) * u_up[i] / u_dw[i]
                + edge0_vertex0;
            Scalar edge1_vertex =
                (edge1_vertex1 - edge1_vertex0) * v_up[i] / v_dw[i]
                + edge1_vertex0;
            rst[i] = edge0_vertex - edge1_vertex;
        }
        return rst;
    }

    // ** this version can return the true x or y or z tolerance of the co-domain **
    // eps is the interval [-eps,eps] we need to check
    // if [-eps,eps] overlap, return true
    // bbox_in_eps tell us if the box is totally in eps box
    // ms is the minimum seperation
__device__ bool evaluate_bbox_one_dimension_vector_return_tolerance(
        Scalar* t_up,
        Scalar* t_dw,
        Scalar* u_up,
        Scalar* u_dw,
        Scalar* v_up,
        Scalar* v_dw,
        const Scalar* a0s,
        const Scalar* a1s,
        const Scalar* b0s,
        const Scalar* b1s,
        const Scalar* a0e,
        const Scalar* a1e,
        const Scalar* b0e,
        const Scalar* b1e,
        const int dimension,
        const bool check_vf,
        const Scalar eps,
        const Scalar ms,
        bool &bbox_in_eps,
        Scalar &tol)
    {

        Scalar* vs;
        int count = 0;
        bbox_in_eps = false;

        if (check_vf)
        {
            vs = function_vf(
                a0s[dimension], a1s[dimension], b0s[dimension], b1s[dimension],
                a0e[dimension], a1e[dimension], b0e[dimension], b1e[dimension],
                t_up, t_dw, u_up, u_dw, v_up, v_dw);
        }
        else
        {
            vs = function_ee(
                a0s[dimension], a1s[dimension], b0s[dimension], b1s[dimension],
                a0e[dimension], a1e[dimension], b0e[dimension], b1e[dimension],
                t_up, t_dw, u_up, u_dw, v_up, v_dw);
        }

        Scalar minv = vs[0], maxv = vs[0];

        for (int i = 1; i < 8; i++)
        {
            if (minv > vs[i])
            {
                minv = vs[i];
            }
            if (maxv < vs[i])
            {
                maxv = vs[i];
            }
        }
        tol = maxv - minv; // this is the real tolerance
        if (minv - ms > eps || maxv + ms < -eps)
            return false;
        if (minv + ms >= -eps && maxv - ms <= eps)
        {
            bbox_in_eps = true;
        }
        return true;
    }