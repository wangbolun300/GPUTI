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




__device__ void compute_face_vertex_tolerance(const CCDdata &data_in,var_wrapper* vars){
    vars->tvars.v.init(data_in.v0s[0], data_in.v0s[1], data_in.v0s[2]);
    vars->tvars.f0.init(data_in.v1s[0], data_in.v1s[1], data_in.v1s[2]);
    vars->tvars.f1.init(data_in.v2s[0], data_in.v2s[1], data_in.v2s[2]);
    vars->tvars.f2.init(data_in.v3s[0], data_in.v3s[1], data_in.v3s[2]);
    vars->tvars.p000 = vars->tvars.v - vars->tvars.f0;
    vars->tvars.p001 = vars->tvars.v - vars->tvars.f2;
    vars->tvars.p011 = vars->tvars.v - (vars->tvars.f1 + vars->tvars.f2 - vars->tvars.f0); 
    vars->tvars.p010 = vars->tvars.v - vars->tvars.f1;
    vars->tvars.v.init(data_in.v0e[0], data_in.v0e[1], data_in.v0e[2]);
    vars->tvars.f0.init(data_in.v1e[0], data_in.v1e[1], data_in.v1e[2]);
    vars->tvars.f1.init(data_in.v2e[0], data_in.v2e[1], data_in.v2e[2]);
    vars->tvars.f2.init(data_in.v3e[0], data_in.v3e[1], data_in.v3e[2]);   
    vars->tvars.p100 = vars->tvars.v - vars->tvars.f0; 
    vars->tvars.p101 = vars->tvars.v - vars->tvars.f2;
    vars->tvars.p111 = vars->tvars.v - (vars->tvars.f1 + vars->tvars.f2 - vars->tvars.f0); 
    vars->tvars.p110 = vars->tvars.v - vars->tvars.f1;
    
    vars->tvars.dl = 3 * max_linf_4(vars->tvars.p000, vars->tvars.p001, vars->tvars.p011, vars->tvars.p010, 
    vars->tvars.p100, vars->tvars.p101, vars->tvars.p111, vars->tvars.p110);
    vars->tvars.edge0_length =
        3 * max_linf_4(vars->tvars.p000, vars->tvars.p100, vars->tvars.p101, vars->tvars.p001, 
        vars->tvars.p010, vars->tvars.p110, vars->tvars.p111, vars->tvars.p011);
    vars->tvars.edge1_length =
        3 * max_linf_4(vars->tvars.p000, vars->tvars.p100, vars->tvars.p110, vars->tvars.p010, 
        vars->tvars.p001, vars->tvars.p101, vars->tvars.p111, vars->tvars.p011);

    vars->out.tol[0] = vars->config.co_domain_tolerance / vars->tvars.dl;
    vars->out.tol[1] = vars->config.co_domain_tolerance / vars->tvars.edge0_length;
    vars->out.tol[2] = vars->config.co_domain_tolerance / vars->tvars.edge1_length;
}

__device__ __host__ void get_numerical_error_vf(
    const CCDdata &data_in,
    var_wrapper* vars)
{

#ifdef GPUTI_USE_DOUBLE_PRECISION
    vars->evars.vffilter = 6.661338147750939e-15;
#else
    vars->evars.vffilter = 3.576279e-06;
#endif
    vars->evars.xmax = fabs(data_in.v0s[0]);
    vars->evars.ymax = fabs(data_in.v0s[1]);
    vars->evars.zmax = fabs(data_in.v0s[2]);

    vars->evars.xmax = max(vars->evars.xmax,fabs(data_in.v1s[0]));
    vars->evars.ymax = max(vars->evars.ymax,fabs(data_in.v1s[1]));
    vars->evars.zmax = max(vars->evars.zmax,fabs(data_in.v1s[2]));
    
    vars->evars.xmax = max(vars->evars.xmax,fabs(data_in.v2s[0]));
    vars->evars.ymax = max(vars->evars.ymax,fabs(data_in.v2s[1]));
    vars->evars.zmax = max(vars->evars.zmax,fabs(data_in.v2s[2]));

    vars->evars.xmax = max(vars->evars.xmax,fabs(data_in.v3s[0]));
    vars->evars.ymax = max(vars->evars.ymax,fabs(data_in.v3s[1]));
    vars->evars.zmax = max(vars->evars.zmax,fabs(data_in.v3s[2]));

    vars->evars.xmax = max(vars->evars.xmax,fabs(data_in.v0e[0]));
    vars->evars.ymax = max(vars->evars.ymax,fabs(data_in.v0e[1]));
    vars->evars.zmax = max(vars->evars.zmax,fabs(data_in.v0e[2]));

    vars->evars.xmax = max(vars->evars.xmax,fabs(data_in.v1e[0]));
    vars->evars.ymax = max(vars->evars.ymax,fabs(data_in.v1e[1]));
    vars->evars.zmax = max(vars->evars.zmax,fabs(data_in.v1e[2]));

    vars->evars.xmax = max(vars->evars.xmax,fabs(data_in.v2e[0]));
    vars->evars.ymax = max(vars->evars.ymax,fabs(data_in.v2e[1]));
    vars->evars.zmax = max(vars->evars.zmax,fabs(data_in.v2e[2]));

    vars->evars.xmax = max(vars->evars.xmax,fabs(data_in.v3e[0]));
    vars->evars.ymax = max(vars->evars.ymax,fabs(data_in.v3e[1]));
    vars->evars.zmax = max(vars->evars.zmax,fabs(data_in.v3e[2]));

    vars->evars.xmax = max(vars->evars.xmax, Scalar(1));
    vars->evars.ymax = max(vars->evars.ymax, Scalar(1));
    vars->evars.zmax = max(vars->evars.zmax, Scalar(1));

    vars->box.err[0] = vars->evars.xmax * vars->evars.xmax * vars->evars.xmax * vars->evars.vffilter;
    vars->box.err[1] = vars->evars.ymax * vars->evars.ymax * vars->evars.ymax * vars->evars.vffilter;
    vars->box.err[2] = vars->evars.zmax * vars->evars.zmax * vars->evars.zmax * vars->evars.vffilter;
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
__device__ void calculate_tuv(var_wrapper *vars){
    if(vars->bp.b[0]==0){// t0
        vars->bp.t=vars->box.current_item.itv[0].first;
    }
    else{// t1
        vars->bp.t=vars->box.current_item.itv[0].second;
    }

    if(vars->bp.b[1]==0){// u0
        vars->bp.u=vars->box.current_item.itv[1].first;
    }
    else{// u1
        vars->bp.u=vars->box.current_item.itv[1].second;
    }

    if(vars->bp.b[2]==0){// v0
        vars->bp.v=vars->box.current_item.itv[2].first;
    }
    else{// v1
        vars->bp.v=vars->box.current_item.itv[2].second;
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

__device__ bool Origin_in_vf_inclusion_function(const CCDdata &data_in, var_wrapper* vars){
    
    Scalar vmin=SCALAR_LIMIT;
    Scalar vmax=-SCALAR_LIMIT;
    Scalar value;
    for(vars->bp.dim=0;vars->bp.dim<3;vars->bp.dim++){
        vmin=SCALAR_LIMIT;
        vmax=-SCALAR_LIMIT;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    vars->bp.b[0] = i;
                    vars->bp.b[1] = j;
                    vars->bp.b[2] = k; //100
                    calculate_tuv(vars);
                    value = calculate_vf(data_in, vars->bp);
                    vmin = min(vmin, value);
                    vmax = max(vmax, value);
                    
                }
            }
        }

        // get the min and max in one dimension
        vars->box.true_tol = max(vars->box.true_tol, vmax - vmin); // this is the real tolerance

        if (vmin > vars->box.err[vars->bp.dim] || vmax < -vars->box.err[vars->bp.dim])
        {
            return false;
        }

        if (vmin < -vars->box.err[vars->bp.dim] || vmax > vars->box.err[vars->bp.dim])
        {
            vars->box.box_in = false;
        }
        
    }
    return true;
}
__device__ void split_dimension(var_wrapper *vars){
    Scalar res[3];
    res[0]=vars->box.widths[0]/vars->out.tol[0];
    res[1]=vars->box.widths[1]/vars->out.tol[1];
    res[2]=vars->box.widths[2]/vars->out.tol[2];
    if(res[0]>=res[1]&&res[0]>=res[2]){
        vars->box.split=0;
    }
    if(res[1]>=res[0]&&res[1]>=res[2]){
        vars->box.split=1;
    }
    if(res[2]>=res[1]&&res[2]>=res[0]){
        vars->box.split=2;
    }
}

__device__ void bisect_vf_and_push(var_wrapper* vars,const CCDdata& data){
    interval_pair halves(vars->box.current_item.itv[vars->box.split]);// bisected
    bool inserted;
    item pushed;
    if (halves.first.first  >= halves.first.second)
    {
        vars->out.overflow_flag = BISECTION_OVERFLOW;
        return;
    }
    if (halves.second.first>= halves.second.second)
    {
        vars->out.overflow_flag = BISECTION_OVERFLOW;
        return;
    }
    if (vars->box.split == 0)// split t interval
    {
        if (data.max_t!=1)
        {
            if (halves.second.first <= data.max_t)
            {
                vars->box.current_item.itv[vars->box.split] = halves.second;
                pushed.itv[0]=vars->box.current_item.itv[0];
                pushed.itv[1]=vars->box.current_item.itv[1];
                pushed.itv[2]=vars->box.current_item.itv[2];
                pushed.level=vars->box.current_item.level + 1;
                inserted = vars->istack.insertKey(pushed);
                if (inserted == false)
                {
                    vars->out.overflow_flag = HEAP_OVERFLOW;
                }
            }

            vars->box.current_item.itv[vars->box.split] = halves.first;
            pushed.itv[0]=vars->box.current_item.itv[0];
            pushed.itv[1]=vars->box.current_item.itv[1];
            pushed.itv[2]=vars->box.current_item.itv[2];
            pushed.level=vars->box.current_item.level + 1;
            inserted = vars->istack.insertKey(pushed);
            if (inserted == false)
            {
                vars->out.overflow_flag = HEAP_OVERFLOW;
            }
        }
        else
        {
            vars->box.current_item.itv[vars->box.split] = halves.second;
            pushed.itv[0]=vars->box.current_item.itv[0];
            pushed.itv[1]=vars->box.current_item.itv[1];
            pushed.itv[2]=vars->box.current_item.itv[2];
            pushed.level=vars->box.current_item.level + 1;
            inserted = vars->istack.insertKey(pushed);
            if (inserted == false)
            {
                vars->out.overflow_flag = HEAP_OVERFLOW;
            }
            vars->box.current_item.itv[vars->box.split] = halves.first;
            pushed.itv[0]=vars->box.current_item.itv[0];
            pushed.itv[1]=vars->box.current_item.itv[1];
            pushed.itv[2]=vars->box.current_item.itv[2];
            pushed.level=vars->box.current_item.level + 1;
            inserted = vars->istack.insertKey(pushed);
            if (inserted == false)
            {
                vars->out.overflow_flag = HEAP_OVERFLOW;
            }
        }
    }

    if (vars->box.split == 1) // split u interval
    {

        if (sum_no_larger_1(halves.second.first, vars->box.current_item.itv[2].first)) // check if u+v<=1
        {

            vars->box.current_item.itv[vars->box.split] = halves.second;
            pushed.itv[0]=vars->box.current_item.itv[0];
            pushed.itv[1]=vars->box.current_item.itv[1];
            pushed.itv[2]=vars->box.current_item.itv[2];
            pushed.level=vars->box.current_item.level + 1;
            // LINENBR 20
            inserted = vars->istack.insertKey(pushed);
            if (inserted == false)
            {
                vars->out.overflow_flag = HEAP_OVERFLOW;
            }
        }

        vars->box.current_item.itv[vars->box.split] = halves.first;
        pushed.itv[0]=vars->box.current_item.itv[0];
        pushed.itv[1]=vars->box.current_item.itv[1];
        pushed.itv[2]=vars->box.current_item.itv[2];
        pushed.level=vars->box.current_item.level + 1;
        inserted = vars->istack.insertKey(pushed);
        if (inserted == false)
        {
            vars->out.overflow_flag = HEAP_OVERFLOW;
        }
    }
    if (vars->box.split == 2) // split v interval
    {
        if (sum_no_larger_1(halves.second.first, vars->box.current_item.itv[1].first))
        {
            vars->box.current_item.itv[vars->box.split] = halves.second;
            pushed.itv[0]=vars->box.current_item.itv[0];
            pushed.itv[1]=vars->box.current_item.itv[1];
            pushed.itv[2]=vars->box.current_item.itv[2];
            pushed.level=vars->box.current_item.level + 1;
            inserted = vars->istack.insertKey(pushed);
            if (inserted == false)
            {
                vars->out.overflow_flag = HEAP_OVERFLOW;
            }
        }

        vars->box.current_item.itv[vars->box.split] = halves.first;
        pushed.itv[0]=vars->box.current_item.itv[0];
        pushed.itv[1]=vars->box.current_item.itv[1];
        pushed.itv[2]=vars->box.current_item.itv[2];
        pushed.level=vars->box.current_item.level + 1;
        inserted = vars->istack.insertKey(pushed);
        if (inserted == false)
        {
            vars->out.overflow_flag = HEAP_OVERFLOW;
        }
    }
}

__device__ bool vertexFaceCCD(const CCDdata &data_in,var_wrapper* vars){
    
    // now when initialized, size is 1 and initialized with [0,1]^3
    compute_face_vertex_tolerance(data_in, vars);

#ifdef CALCULATE_ERROR_BOUND
    get_numerical_error_vf(data_in, vars);
#else
    vars->box.err[0] = vars->config.err_in[0];
    vars->box.err[1] = vars->config.err_in[1];
    vars->box.err[2] = vars->config.err_in[2];
#endif
//return true;
    vars->out.output_tolerance = vars->config.co_domain_tolerance;
    vars->out.toi=SCALAR_LIMIT;
    vars->out.overflow_flag=NO_OVERFLOW;

vars->refine = 0;
    // temp_toi is to catch the first toi of each level
vars->temp_toi = SCALAR_LIMIT;
vars->skip_toi =SCALAR_LIMIT;
vars->use_skip = false; // when tolerance is small enough or when box in epsilon, this is activated.
vars->current_level = -2; // in the begining, current_level != level
vars->box_in_level = -2;  // this checks if all the boxes before this
// level < tolerance. only true, we can return when we find one overlaps eps box and smaller than tolerance or eps-box
vars->this_level_less_tol = true;
vars->find_level_root = false;

vars->temp_output_tolerance = vars->config.co_domain_tolerance;
    
    vars->istack.initialize();
    while (!(vars->istack.empty()))
    {
        
        if (vars->out.overflow_flag != NO_OVERFLOW)
        {
            break;
        }
        
        //LINENBR 6
        vars->box.current_item = vars->istack.extractMin();// get the level and the intervals
        
        // if this box is later than TOI_SKIP in time, we can skip this one.
        // TOI_SKIP is only updated when the box is small enough or totally contained in eps-box
        if (vars->box.current_item.itv[0].first>=vars->skip_toi)
        {
            continue;
        }
        
        if (vars->box_in_level != vars->box.current_item.level)
        { // before check a new level, set this_level_less_tol=true
            vars->box_in_level = vars->box.current_item.level;
            vars->this_level_less_tol = true;
        }
        // LINENBR 8
        vars->refine++;
        vars->zero_in =
            Origin_in_vf_inclusion_function(data_in,vars);
        // return true;// REGSCOUNT 100
        
        if (!vars->zero_in)
            continue;

        // get the width of the box
        vars->box.widths[0] = vars->box.current_item.itv[0].second - vars->box.current_item.itv[0].first;
        vars->box.widths[1] = vars->box.current_item.itv[1].second - vars->box.current_item.itv[1].first;
        vars->box.widths[2] = vars->box.current_item.itv[2].second - vars->box.current_item.itv[2].first;
                
        // LINENBR 15, 16
        // Condition 1, stopping condition on t, u and v is satisfied. this is useless now since we have condition 2
        vars->condition = vars->box.widths[0] <= vars->out.tol[0] && vars->box.widths[1] <= vars->out.tol[1] && vars->box.widths[2] <= vars->out.tol[2];
        if(vars->condition){
            vars->out.toi=vars->box.current_item.itv[0].first;
            return true;
        }
        // Condition 2, zero_in = true, box inside eps-box and in this level,
        // no box whose zero_in is true but box size larger than tolerance, can return
        vars->condition = vars->box.box_in && vars->this_level_less_tol;
        if(vars->condition){
            vars->out.toi=vars->box.current_item.itv[0].first;
            return true;
        }

        vars->tol_condition = vars->box.true_tol <= vars->config.co_domain_tolerance;
        if (!vars->tol_condition)
        {
            vars->this_level_less_tol = false;
            // this level has at least one box whose size > tolerance, thus we
            // cannot directly return if find one box whose size < tolerance or box-in
        }

        // Condition 3, in this level, we find a box that zero-in and size < tolerance.
        // and no other boxes whose zero-in is true in this level before this one is larger than tolerance, can return
        vars->condition = vars->this_level_less_tol;
        if(vars->condition){
            vars->out.toi=vars->box.current_item.itv[0].first;
            return true;
        }

        // This is for early termination, finding the earlist root of this level in case of early termination happens
        if (vars->current_level != vars->box.current_item.level)
        {
            // LINENBR 22
            vars->current_level = vars->box.current_item.level;
            vars->find_level_root = false;
        }
        if (!vars->find_level_root)
        {
            // LINENBR 11
            // this is the first toi of this level
            vars->temp_toi = vars->box.current_item.itv[0].first;
            // if the real tolerance is larger than input, use the real one;
            // if the real tolerance is smaller than input, use input
            vars->temp_output_tolerance = max(vars->box.true_tol,vars->config.co_domain_tolerance);
            vars->find_level_root =true; // this ensures always find the earlist root
        }


        // if this box is small enough, or inside of eps-box, then just continue,
        // but we need to record the collision time
        if (vars->tol_condition || vars->box.box_in )
        {
            if(vars->box.current_item.itv[0].first<vars->skip_toi)
            {
                vars->skip_toi=vars->box.current_item.itv[0].first;
            }
            vars->use_skip = true;
            continue;
        }
        split_dimension(vars);
        bisect_vf_and_push(vars,data_in);
    }
    if (vars->out.overflow_flag != NO_OVERFLOW)
    {
        vars->out.toi = vars->temp_toi;
        vars->out.output_tolerance = vars->temp_output_tolerance;
        return true;
    }

    if (vars->use_skip)
    {
        vars->out.toi = vars->skip_toi;

        return true;
    }
    return false;
}


