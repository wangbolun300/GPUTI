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


__device__ void compute_face_vertex_tolerance(const CCDdata &data_in,const CCDConfig& config, CCDOut* out){
    Scalar p000[3], p001[3], p011[3], p010[3], p100[3], p101[3], p111[3], p110[3];
    for(int i=0;i<3;i++){
        p000[i] = data_in.v0s[i] - data_in.v1s[i]; 
        p001[i] = data_in.v0s[i] - data_in.v3s[i];
        p011[i] = data_in.v0s[i] - (data_in.v2s[i] + data_in.v3s[i] - data_in.v1s[i]); 
        p010[i] = data_in.v0s[i] - data_in.v2s[i];
        p100[i] = data_in.v0e[i] - data_in.v1e[i]; 
        p101[i] = data_in.v0e[i] - data_in.v3e[i];
        p111[i] = data_in.v0e[i] - (data_in.v2e[i] + data_in.v3e[i] - data_in.v1e[i]); 
        p110[i] = data_in.v0e[i] - data_in.v2e[i];
    }
    Scalar dl=0;
    for(int i=0;i<3;i++){
        dl=max(dl,fabs(p100[i]-p000[i]));
        dl=max(dl,fabs(p101[i]-p001[i])); 
        dl=max(dl,fabs(p111[i]-p011[i]));
        dl=max(dl,fabs(p110[i]-p010[i]));
    }
    dl*=3;
    out->tol[0] = config.co_domain_tolerance / dl;

    dl=0;
    for(int i=0;i<3;i++){
        dl=max(dl,fabs(p010[i]-p000[i]));
        dl=max(dl,fabs(p110[i]-p100[i])); 
        dl=max(dl,fabs(p111[i]-p101[i]));
        dl=max(dl,fabs(p011[i]-p001[i]));
    }
    dl*=3;
    out->tol[1] = config.co_domain_tolerance / dl;
    
    dl=0;
    for(int i=0;i<3;i++){
        dl=max(dl,fabs(p001[i]-p000[i]));
        dl=max(dl,fabs(p101[i]-p100[i])); 
        dl=max(dl,fabs(p111[i]-p110[i]));
        dl=max(dl,fabs(p011[i]-p010[i]));
    }
    dl*=3;
    out->tol[2] = config.co_domain_tolerance / dl;
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

__device__ bool Origin_in_vf_inclusion_function(const CCDdata &data_in, BoxCompute& box, CCDOut* out){
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
__device__ void split_dimension(CCDOut* out,BoxCompute& box){
    Scalar res[3];
    res[0]=box.widths[0]/out->tol[0];
    res[1]=box.widths[1]/out->tol[1];
    res[2]=box.widths[2]/out->tol[2];
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

__device__ void bisect_vf_and_push(BoxCompute& box,const CCDConfig& config, MinHeap& istack,CCDOut* out){
    interval_pair halves(box.current_item.itv[box.split]);// bisected
    bool inserted;
    if (halves.first.first  >= halves.first.second)
    {
        out->overflow_flag = BISECTION_OVERFLOW;
        return;
    }
    if (halves.second.first>= halves.second.second)
    {
        out->overflow_flag = BISECTION_OVERFLOW;
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
                    out->overflow_flag = HEAP_OVERFLOW;
                }
            }

            box.current_item.itv[box.split] = halves.first;
            inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
            if (inserted == false)
            {
                out->overflow_flag = HEAP_OVERFLOW;
            }
        }
        else
        {
            box.current_item.itv[box.split] = halves.second;
            inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
            if (inserted == false)
            {
                out->overflow_flag = HEAP_OVERFLOW;
            }
            box.current_item.itv[box.split] = halves.first;
            inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
            if (inserted == false)
            {
                out->overflow_flag = HEAP_OVERFLOW;
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
                out->overflow_flag = HEAP_OVERFLOW;
            }
        }

        box.current_item.itv[box.split] = halves.first;
        inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
        if (inserted == false)
        {
            out->overflow_flag = HEAP_OVERFLOW;
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
                out->overflow_flag = HEAP_OVERFLOW;
            }
        }

        box.current_item.itv[box.split] = halves.first;
        inserted = istack.insertKey(item(box.current_item.itv, box.current_item.level + 1));
        if (inserted == false)
        {
            out->overflow_flag = HEAP_OVERFLOW;
        }
    }
}

__device__ void vertexFaceCCD(CCDdata *data,const CCDConfig& config, CCDOut* out){
    CCDdata data_in;
    for (int i = 0; i < 3; i++)
    {
        data_in.v0s[i] = data->v0s[i];
        data_in.v1s[i] = data->v1s[i];
        data_in.v2s[i] = data->v2s[i];
        data_in.v3s[i] = data->v3s[i];
        data_in.v0e[i] = data->v0e[i];
        data_in.v1e[i] = data->v1e[i];
        data_in.v2e[i] = data->v2e[i];
        data_in.v3e[i] = data->v3e[i];
    }
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

    out->output_tolerance = config.co_domain_tolerance;

    // this is used to catch the tolerance for each level
    Scalar temp_output_tolerance = config.co_domain_tolerance;
    // LINENBR 2
    int refine = 0;
    // temp_toi is to catch the first toi of each level
    Scalar temp_toi = SCALAR_LIMIT;
    Scalar skip_toi =SCALAR_LIMIT;
    
    bool use_skip = false; // when tolerance is small enough or when box in epsilon, this is activated.
    int current_level = -2; // in the begining, current_level != level
    int box_in_level = -2;  // this checks if all the boxes before this
    // level < tolerance. only true, we can return when we find one overlaps eps box and smaller than tolerance or eps-box
    bool this_level_less_tol = true;
    bool find_level_root = false;

    while (!istack.empty())
    {
        if (out->overflow_flag != NO_OVERFLOW)
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
        
        if (!zero_in)
            continue;

        // get the width of the box
        box.widths[0] = box.current_item.itv[0].second - box.current_item.itv[0].first;
        box.widths[1] = box.current_item.itv[1].second - box.current_item.itv[1].first;
        box.widths[2] = box.current_item.itv[2].second - box.current_item.itv[2].first;
                
        // LINENBR 15, 16
        // Condition 1, stopping condition on t, u and v is satisfied. this is useless now since we have condition 2
        bool condition = box.widths[0] <= out->tol[0] && box.widths[1] <= out->tol[1] && box.widths[2] <= out->tol[2];
        if(condition){
            out->toi=box.current_item.itv[0].first;
            out->result=true;
            return;
        }
        // Condition 2, zero_in = true, box inside eps-box and in this level,
        // no box whose zero_in is true but box size larger than tolerance, can return
        condition = box.box_in && this_level_less_tol;
        if(condition){
            out->toi=box.current_item.itv[0].first;
            out->result= true;
            return;
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
            out->toi=box.current_item.itv[0].first;
            out->result=true;
            return;
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
            out->overflow_flag = ITERATION_OVERFLOW;
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
    if (out->overflow_flag != NO_OVERFLOW)
    {
        out->toi = temp_toi;
        out->output_tolerance = temp_output_tolerance;
        out->result=true;
        return;
    }

    if (use_skip)
    {
        out->toi = skip_toi;
        out->result=true;
        return;
    }
    out->result=false;
    return;
}


