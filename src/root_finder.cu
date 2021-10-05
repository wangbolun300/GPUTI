#include <gputi/root_finder.h>
#include <gputi/queue.h>
#include <iostream>

CCDdata array_to_ccd(std::array<std::array<Scalar, 3>, 8> a, bool is_edge)
{
    CCDdata data;
#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        data.v0s[i] = a[0][i];
        data.v1s[i] = a[1][i];
        data.v2s[i] = a[2][i];
        data.v3s[i] = a[3][i];
        data.v0e[i] = a[4][i];
        data.v1e[i] = a[5][i];
        data.v2e[i] = a[6][i];
        data.v3e[i] = a[7][i];
    }
    return data;
}
__device__ __host__ VectorMax3d::VectorMax3d(Scalar a, Scalar b, Scalar c)
{
    v[0] = a;
    v[1] = b;
    v[2] = c;
}
__device__ __host__ void VectorMax3d::init(Scalar a, Scalar b, Scalar c){
    v[0] = a;
    v[1] = b;
    v[2] = c;
}


__device__ Singleinterval::Singleinterval(const Scalar& f, const Scalar& s)
{
    first = f;
    second = s;
}

// this function do the bisection
__device__ interval_pair::interval_pair(const Singleinterval& itv){
    Scalar c=(itv.first+itv.second)/2;
    first.first=itv.first;
    first.second=c;
    second.first=c;
    second.second=itv.second;
}

// TODO need to calculate error bound
__device__ bool sum_no_larger_1(const Scalar &num1, const Scalar &num2)
{
    if(num1+num2<=1){
        return true;
    }
    return false;
}


__device__ Scalar max_linf_dist(const VectorMax3d &p1, const VectorMax3d &p2)
{
    Scalar r = 0;
    r = max(r, fabs(p1.v[0] - p2.v[0]));
    r = max(r, fabs(p1.v[1] - p2.v[1]));
    r = max(r, fabs(p1.v[2] - p2.v[2]));
    return r;
}

__device__ Scalar max_linf_4(
    const VectorMax3d &p1,
    const VectorMax3d &p2,
    const VectorMax3d &p3,
    const VectorMax3d &p4,
    const VectorMax3d &p1e,
    const VectorMax3d &p2e,
    const VectorMax3d &p3e,
    const VectorMax3d &p4e)
{
    Scalar r = 0; //,temp = 0, temp2=0, temp3 = 0, temp4=0;
    r = max(r, max_linf_dist(p4e, p4));
    r = max(r, max_linf_dist(p3e, p3));
    r = max(r, max_linf_dist(p2e, p2));
    return max(r, max_linf_dist(p1e, p1));
}




__device__ void compute_face_vertex_tolerance(const CCDdata &data_in,const CCDConfig& config, CCDOut& out){
    VectorMax3d v(data_in.v0s[0], data_in.v0s[1], data_in.v0s[2]);
    VectorMax3d f0(data_in.v1s[0], data_in.v1s[1], data_in.v1s[2]);
    VectorMax3d f1(data_in.v2s[0], data_in.v2s[1], data_in.v2s[2]);
    VectorMax3d f2(data_in.v3s[0], data_in.v3s[1], data_in.v3s[2]);
    VectorMax3d p000 = v - f0, p001 = v - f2,
                p011 = v - (f1 + f2 - f0), p010 = v - f1;
    v.init(data_in.v0e[0], data_in.v0e[1], data_in.v0e[2]);
    f0.init(data_in.v1e[0], data_in.v1e[1], data_in.v1e[2]);
    f1.init(data_in.v2e[0], data_in.v2e[1], data_in.v2e[2]);
    f2.init(data_in.v3e[0], data_in.v3e[1], data_in.v3e[2]);   
    VectorMax3d p100 = v - f0, p101 = v - f2,
                p111 = v - (f1 + f2 - f0), p110 = v - f1;
    
    Scalar dl = 3 * max_linf_4(p000, p001, p011, p010, p100, p101, p111, p110);
    Scalar edge0_length =
        3 * max_linf_4(p000, p100, p101, p001, p010, p110, p111, p011);
    Scalar edge1_length =
        3 * max_linf_4(p000, p100, p110, p010, p001, p101, p111, p011);

    out.tol[0] = config.co_domain_tolerance / dl;
    out.tol[1] = config.co_domain_tolerance / edge0_length;
    out.tol[2] = config.co_domain_tolerance / edge1_length;
}

__device__ __host__ void get_numerical_error_vf(
    const CCDdata &data_in,
    BoxCompute &box)
{
    Scalar vffilter;

#ifdef GPUTI_USE_DOUBLE_PRECISION
    vffilter = 6.661338147750939e-15;
#else
    vffilter = 3.576279e-06;
#endif
    Scalar xmax = fabs(data_in.v0s[0]);
    Scalar ymax = fabs(data_in.v0s[1]);
    Scalar zmax = fabs(data_in.v0s[2]);

    xmax = max(xmax,fabs(data_in.v1s[0]));
    ymax = max(ymax,fabs(data_in.v1s[1]));
    zmax = max(zmax,fabs(data_in.v1s[2]));
    
    xmax = max(xmax,fabs(data_in.v2s[0]));
    ymax = max(ymax,fabs(data_in.v2s[1]));
    zmax = max(zmax,fabs(data_in.v2s[2]));

    xmax = max(xmax,fabs(data_in.v3s[0]));
    ymax = max(ymax,fabs(data_in.v3s[1]));
    zmax = max(zmax,fabs(data_in.v3s[2]));

    xmax = max(xmax,fabs(data_in.v0e[0]));
    ymax = max(ymax,fabs(data_in.v0e[1]));
    zmax = max(zmax,fabs(data_in.v0e[2]));

    xmax = max(xmax,fabs(data_in.v1e[0]));
    ymax = max(ymax,fabs(data_in.v1e[1]));
    zmax = max(zmax,fabs(data_in.v1e[2]));

    xmax = max(xmax,fabs(data_in.v2e[0]));
    ymax = max(ymax,fabs(data_in.v2e[1]));
    zmax = max(zmax,fabs(data_in.v2e[2]));

    xmax = max(xmax,fabs(data_in.v3e[0]));
    ymax = max(ymax,fabs(data_in.v3e[1]));
    zmax = max(zmax,fabs(data_in.v3e[2]));

    xmax = max(xmax, Scalar(1));
    ymax = max(ymax, Scalar(1));
    zmax = max(zmax, Scalar(1));

    box.err[0] = xmax * xmax * xmax * vffilter;
    box.err[1] = ymax * ymax * ymax * vffilter;
    box.err[2] = zmax * zmax * zmax * vffilter;
    return;
}
// Singleinterval *paras,
//     const Scalar *a0s,
//     const Scalar *a1s,
//     const Scalar *b0s,
//     const Scalar *b1s,
//     const Scalar *a0e,
//     const Scalar *a1e,
//     const Scalar *b0e,
//     const Scalar *b1e,
//     const bool check_vf,
//     const Scalar *box,
//     const Scalar ms,
//     bool &box_in_eps,
//     Scalar *tolerance)
__device__ void BoxPrimatives::calculate_tuv(const BoxCompute& box){
    if(b[0]==0){// t0
        t=box.current_item.itv[0].first;
    }
    else{// t1
        t=box.current_item.itv[0].second;
    }

    if(b[1]==0){// u0
        u=box.current_item.itv[1].first;
    }
    else{// u1
        u=box.current_item.itv[1].second;
    }

    if(b[2]==0){// v0
        v=box.current_item.itv[2].first;
    }
    else{// v1
        v=box.current_item.itv[2].second;
    }
}
__device__ Scalar calculate_vf(const CCDdata &data_in, const BoxPrimatives& bp){
    Scalar v, pt, t0, t1, t2;
    v = (data_in.v0e[bp.dim] - data_in.v0s[bp.dim]) * bp.t + data_in.v0s[bp.dim];
        t0 = (data_in.v1e[bp.dim] - data_in.v1s[bp.dim]) * bp.t + data_in.v1s[bp.dim];
        t1 = (data_in.v2e[bp.dim] - data_in.v2s[bp.dim]) * bp.t + data_in.v2s[bp.dim];
        t2 = (data_in.v3e[bp.dim] - data_in.v3s[bp.dim]) * bp.t + data_in.v3s[bp.dim];
        pt = (t1 - t0) * bp.u + (t2 - t0) * bp.v + t0;
        return (v - pt);
}

__device__ bool Origin_in_vf_inclusion_function(const CCDdata &data_in, BoxCompute& box, CCDOut& out){
    BoxPrimatives bp;
    Scalar vmin=SCALAR_LIMIT;
    Scalar vmax=-SCALAR_LIMIT;
    Scalar value;
    for(bp.dim=0;bp.dim<3;bp.dim++){
        vmin=SCALAR_LIMIT;
        vmax=-SCALAR_LIMIT;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    bp.b[0] = i;
                    bp.b[1] = j;
                    bp.b[2] = k; //100
                    bp.calculate_tuv(box);
                    value = calculate_vf(data_in, bp);
                    vmin = min(vmin, value);
                    vmax = max(vmax, value);
                    
                }
            }
        }

        // get the min and max in one dimension
        box.true_tol = max(box.true_tol, vmax - vmin); // this is the real tolerance

        if (vmin > box.err[bp.dim] || vmax < -box.err[bp.dim])
        {
            return false;
        }

        if (vmin < -box.err[bp.dim] || vmax > box.err[bp.dim])
        {
            box.box_in = false;
        }
        
    }
    return true;
}
__device__ void split_dimension(const CCDOut& out,BoxCompute& box){
    Scalar res[3];
    res[0]=box.widths[0]/out.tol[0];
    res[1]=box.widths[1]/out.tol[1];
    res[2]=box.widths[2]/out.tol[2];
    if(res[0]>=res[1]&&res[0]>=res[2]){
        box.split=0;
    }
    if(res[1]>=res[0]&&res[1]>=res[2]){
        box.split=1;
    }
    if(res[2]>=res[1]&&res[2]>=res[0]){
        box.split=2;
    }
}

__device__ void bisect_vf_and_push(BoxCompute& box,const CCDConfig& config, MinHeap& istack,CCDOut& out){
    interval_pair halves(box.current_item.itv[box.split]);// bisected
    bool inserted;
    if (halves.first.first  >= halves.first.second)
    {
        out.overflow_flag = BISECTION_OVERFLOW;
        return;
    }
    if (halves.second.first>= halves.second.second)
    {
        out.overflow_flag = BISECTION_OVERFLOW;
        return;
    }
    if (box.split == 0)// split t interval
    {
        if (config.max_t!=1)
        {
            if (halves.second.first <= config.max_t)
            {
                box.current_item.itv[box.split] = halves.second;
                inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
                if (inserted == false)
                {
                    out.overflow_flag = HEAP_OVERFLOW;
                }
            }

            box.current_item.itv[box.split] = halves.first;
            inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
            if (inserted == false)
            {
                out.overflow_flag = HEAP_OVERFLOW;
            }
        }
        else
        {
            box.current_item.itv[box.split] = halves.second;
            inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
            if (inserted == false)
            {
                out.overflow_flag = HEAP_OVERFLOW;
            }
            box.current_item.itv[box.split] = halves.first;
            inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
            if (inserted == false)
            {
                out.overflow_flag = HEAP_OVERFLOW;
            }
        }
    }

    if (box.split == 1) // split u interval
    {

        if (sum_no_larger_1(halves.second.first, box.current_item.itv[2].first)) // check if u+v<=1
        {

            box.current_item.itv[box.split] = halves.second;
            // LINENBR 20
            inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
            if (inserted == false)
            {
                out.overflow_flag = HEAP_OVERFLOW;
            }
        }

        box.current_item.itv[box.split] = halves.first;
        inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
        if (inserted == false)
        {
            out.overflow_flag = HEAP_OVERFLOW;
        }
    }
    if (box.split == 2) // split v interval
    {
        if (sum_no_larger_1(halves.second.first, box.current_item.itv[1].first))
        {
            box.current_item.itv[box.split] = halves.second;
            inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
            if (inserted == false)
            {
                out.overflow_flag = HEAP_OVERFLOW;
            }
        }

        box.current_item.itv[box.split] = halves.first;
        inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
        if (inserted == false)
        {
            out.overflow_flag = HEAP_OVERFLOW;
        }
    }
}

__device__ bool vertexFaceCCD(const CCDdata &data_in,const CCDConfig& config, CCDOut& out){
    
    MinHeap istack;// now when initialized, size is 1 and initialized with [0,1]^3
    compute_face_vertex_tolerance(data_in, config, out);
    BoxCompute box;

#ifdef CALCULATE_ERROR_BOUND
    get_numerical_error_vf(data_in, box);
#else
    box.err[0] = config.err_in[0];
    box.err[1] = config.err_in[1];
    box.err[2] = config.err_in[2];
#endif

    out.output_tolerance = config.co_domain_tolerance;

    // this is used to catch the tolerance for each level
    Scalar temp_output_tolerance = config.co_domain_tolerance;
    // LINENBR 2
    int refine = 0;
    // temp_toi is to catch the first toi of each level
    Scalar temp_toi = out.toi;
    Scalar skip_toi =out.toi;
    
    bool use_skip = false; // when tolerance is small enough or when box in epsilon, this is activated.
    int current_level = -2; // in the begining, current_level != level
    int box_in_level = -2;  // this checks if all the boxes before this
    // level < tolerance. only true, we can return when we find one overlaps eps box and smaller than tolerance or eps-box
    bool this_level_less_tol = true;
    bool find_level_root = false;

    while (!istack.empty())
    {
        if (out.overflow_flag != NO_OVERFLOW)
        {
            break;
        }

        //LINENBR 6
        box.current_item = istack.extractMin();// get the level and the intervals
        
        // if this box is later than TOI_SKIP in time, we can skip this one.
        // TOI_SKIP is only updated when the box is small enough or totally contained in eps-box
        if (box.current_item.itv[0].first>=skip_toi)
        {
            continue;
        }
        if (box_in_level != box.current_item.level)
        { // before check a new level, set this_level_less_tol=true
            box_in_level = box.current_item.level;
            this_level_less_tol = true;
        }
        // LINENBR 8
        refine++;
        bool zero_in =
            Origin_in_vf_inclusion_function(data_in,box, out);
        //return zero_in;// REGSCOUNT 100
        
        if (!zero_in)
            continue;

        // get the width of the box
        box.widths[0] = box.current_item.itv[0].second - box.current_item.itv[0].first;
        box.widths[1] = box.current_item.itv[1].second - box.current_item.itv[1].first;
        box.widths[2] = box.current_item.itv[2].second - box.current_item.itv[2].first;
                
        // LINENBR 15, 16
        // Condition 1, stopping condition on t, u and v is satisfied. this is useless now since we have condition 2
        bool condition = box.widths[0] <= out.tol[0] && box.widths[1] <= out.tol[1] && box.widths[2] <= out.tol[2];
        if(condition){
            out.toi=box.current_item.itv[0].first;
            return true;
        }
        // Condition 2, zero_in = true, box inside eps-box and in this level,
        // no box whose zero_in is true but box size larger than tolerance, can return
        condition = box.box_in && this_level_less_tol;
        if(condition){
            out.toi=box.current_item.itv[0].first;
            return true;
        }

        bool tol_condition = box.true_tol <= config.co_domain_tolerance;
        if (!tol_condition)
        {
            this_level_less_tol = false;
            // this level has at least one box whose size > tolerance, thus we
            // cannot directly return if find one box whose size < tolerance or box-in
        }

        // Condition 3, in this level, we find a box that zero-in and size < tolerance.
        // and no other boxes whose zero-in is true in this level before this one is larger than tolerance, can return
        condition = this_level_less_tol;
        if(condition){
            out.toi=box.current_item.itv[0].first;
            return true;
        }

        // This is for early termination, finding the earlist root of this level in case of early termination happens
        if (current_level != box.current_item.level)
        {
            // LINENBR 22
            current_level = box.current_item.level;
            find_level_root = false;
        }
        if (!find_level_root)
        {
            // LINENBR 11
            // this is the first toi of this level
            temp_toi = box.current_item.itv[0].first;
            // if the real tolerance is larger than input, use the real one;
            // if the real tolerance is smaller than input, use input
            temp_output_tolerance = max(box.true_tol,config.co_domain_tolerance);
            find_level_root =true; // this ensures always find the earlist root
        }

        // LINENBR 12
        if (refine > config.max_itr)
        {
            out.overflow_flag = ITERATION_OVERFLOW;
            break;
        }

        // if this box is small enough, or inside of eps-box, then just continue,
        // but we need to record the collision time
        if (tol_condition || box.box_in )
        {
            if(box.current_item.itv[0].first<skip_toi)
            {
                skip_toi=box.current_item.itv[0].first;
            }
            use_skip = true;
            continue;
        }
        split_dimension(out,box);
        bisect_vf_and_push(box,config, istack,out);
    }
    if (out.overflow_flag != NO_OVERFLOW)
    {
        out.toi = temp_toi;
        out.output_tolerance = temp_output_tolerance;
        return true;
    }

    if (use_skip)
    {
        out.toi = skip_toi;

        return true;
    }
    return false;
}


