#include <gputi/root_finder.h>
#include <gputi/queue.h>
#include <iostream>

__device__ __host__ void fabs(const Scalar& a, Scalar&result){
    if(a>=0){
        result=a;
    }
    else{
        result=-a;
    }
}
__device__ void VecSum(VectorMax3d& a, VectorMax3d& b, VectorMax3d& res){
    res.v[0]=a.v[0]+b.v[0];
    res.v[1]=a.v[1]+b.v[1];
    res.v[2]=a.v[2]+b.v[2];
}

__device__ void VecMinus( VectorMax3d& a,  VectorMax3d& b,  VectorMax3d& res){
    res.v[0]=a.v[0]-b.v[0];
    res.v[1]=a.v[1]-b.v[1];
    res.v[2]=a.v[2]-b.v[2];
}
void print_vector(Scalar* v, int size){
    for(int i=0;i<size;i++){
        std::cout<<v[i]<<",";
    }
    std::cout<<std::endl;
}
void print_vector(int* v, int size){
    for(int i=0;i<size;i++){
        std::cout<<v[i]<<",";
    }
    std::cout<<std::endl;
}

CCDdata array_to_ccd(std::array<std::array<Scalar, 3>, 8> a, bool is_edge)
{
    CCDdata data;
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
    data.is_edge = is_edge;
    return data;
}
__device__ __host__ void VectorMax3d::init(const Scalar &a, const Scalar &b, const Scalar &c)
{
    v[0] = a;
    v[1] = b;
    v[2] = c;
}
__device__ __host__ VectorMax3d::VectorMax3d(const Scalar &a, const Scalar &b, const Scalar &c)
{
    v[0] = a;
    v[1] = b;
    v[2] = c;
}

__device__ long reduction(const long n, long &result)
{
    int t = 0;
    int newn = n;
    while (newn % 2 == 0)
    {
        newn = newn / 2;
        t++;
    }
    result = newn;
    return t;
}

__device__ Singleinterval::Singleinterval(Numccd f, Numccd s)
{
    first = f;
    second = s;
}
__device__ interval_pair::interval_pair(const Singleinterval &a, const Singleinterval &b)
{
    // interval_cp(a,first);
    // interval_cp(b,second);
    first = a;
    second = b;
}

//TODO add overflow checks for these basis functions
// calculate a*(2^b)
__device__ long power(const long a, const int b) { return a << b; }
__device__ Scalar Numccd2double(const Numccd &n)
{
    Scalar r = Scalar(n.first) / power(1, n.second);
    return r;
}
__device__ Numccd::Numccd(const long a, const int b)
{
    first = a;
    second = b;
}
__device__ bool sum_no_larger_1(const Numccd &num1, const Numccd &num2)
{
    long k1 = num1.first;
    int n1 = num1.second;
    long k2 = num2.first;
    int n2 = num2.second;
    long k;
    int n;
    if (n1 == n2)
    {
        k = k1 + k2;
        n = n1;
    }
    if (n1 < n2)
    {
        k = power(1, n2 - n1) * k1 + k2;
        n = n2;
    }
    if (n1 > n2)
    {
        k = power(1, n1 - n2) * k2 + k1;
        n = n1;
    }
    if (k > power(1, n))
        return false;
    else
        return true;
}
__device__ bool less_than(const Numccd &num1, const Numccd &num2)
{
    long k1 = num1.first;
    int n1 = num1.second;
    long k2 = num2.first;
    int n2 = num2.second;

    if (n1 < n2)
    {
        k1 = power(1, n2 - n1) * k1;
    }
    if (n1 > n2)
    {
        k2 = power(1, n1 - n2) * k2;
    }
    if (k1 < k2)
        return true;
    return false;
}
__device__ bool interval_overlap_region(
    const Singleinterval &itv, const Scalar r1, const Scalar r2)
{
    Scalar b1 = Numccd2double(itv.first);
    Scalar b2 = Numccd2double(itv.second);
    if (b2 < r1 || b1 > r2)
        return false;
    return true;
}

__device__ VectorMax3d width(const Singleinterval *x)
{
    VectorMax3d w;
    for (int i = 0; i < 3; i++)
    {
        w.v[i] = // 0.1;
            (Numccd2double(x[i].second) - Numccd2double(x[i].first));
    }
    return w;
}

__device__ void convert_tuv_to_array(
    const Singleinterval *itv0, Singleinterval *itv1, Singleinterval *itv2,
    Scalar *t_up,
    Scalar *t_dw,
    Scalar *u_up,
    Scalar *u_dw,
    Scalar *v_up,
    Scalar *v_dw)
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

    t_up[0] = t0_up;
    t_up[1] = t0_up;
    t_up[2] = t0_up;
    t_up[3] = t0_up;
    t_up[4] = t1_up;
    t_up[5] = t1_up;
    t_up[6] = t1_up;
    t_up[7] = t1_up;
    t_dw[0] = t0_dw;
    t_dw[1] = t0_dw;
    t_dw[2] = t0_dw;
    t_dw[3] = t0_dw;
    t_dw[4] = t1_dw;
    t_dw[5] = t1_dw;
    t_dw[6] = t1_dw;
    t_dw[7] = t1_dw;
    u_up[0] = u0_up;
    u_up[1] = u0_up;
    u_up[2] = u1_up;
    u_up[3] = u1_up;
    u_up[4] = u0_up;
    u_up[5] = u0_up;
    u_up[6] = u1_up;
    u_up[7] = u1_up;
    u_dw[0] = u0_dw;
    u_dw[1] = u0_dw;
    u_dw[2] = u1_dw;
    u_dw[3] = u1_dw;
    u_dw[4] = u0_dw;
    u_dw[5] = u0_dw;
    u_dw[6] = u1_dw;
    u_dw[7] = u1_dw;
    v_up[0] = v0_up;
    v_up[1] = v1_up;
    v_up[2] = v0_up;
    v_up[3] = v1_up;
    v_up[4] = v0_up;
    v_up[5] = v1_up;
    v_up[6] = v0_up;
    v_up[7] = v1_up;
    v_dw[0] = v0_dw;
    v_dw[1] = v1_dw;
    v_dw[2] = v0_dw;
    v_dw[3] = v1_dw;
    v_dw[4] = v0_dw;
    v_dw[5] = v1_dw;
    v_dw[6] = v0_dw;
    v_dw[7] = v1_dw;
}

__device__ void function_vf(
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
    const Scalar *v_dw,
    Scalar *rst)
{
    
    for (int i = 0; i < 8; i++)
    {
        Scalar v = (ve - vs) * t_up[i] / t_dw[i] + vs;
        Scalar t0 = (t0e - t0s) * t_up[i] / t_dw[i] + t0s;
        Scalar t1 = (t1e - t1s) * t_up[i] / t_dw[i] + t1s;
        Scalar t2 = (t2e - t2s) * t_up[i] / t_dw[i] + t2s;
        Scalar pt = (t1 - t0) * u_up[i] / u_dw[i] + (t2 - t0) * v_up[i] / v_dw[i] + t0;
        rst[i] = v - pt;
    }
}

__device__ void function_ee(
    const Scalar &a0s,
    const Scalar &a1s,
    const Scalar &b0s,
    const Scalar &b1s,
    const Scalar &a0e,
    const Scalar &a1e,
    const Scalar &b0e,
    const Scalar &b1e,
    const Scalar *t_up,
    const Scalar *t_dw,
    const Scalar *u_up,
    const Scalar *u_dw,
    const Scalar *v_up,
    const Scalar *v_dw,
    Scalar *rst)
{

    for (int i = 0; i < 8; i++)
    {
        Scalar edge0_vertex0 = (a0e - a0s) * t_up[i] / t_dw[i] + a0s;
        Scalar edge0_vertex1 = (a1e - a1s) * t_up[i] / t_dw[i] + a1s;
        Scalar edge1_vertex0 = (b0e - b0s) * t_up[i] / t_dw[i] + b0s;
        Scalar edge1_vertex1 = (b1e - b1s) * t_up[i] / t_dw[i] + b1s;

        Scalar edge0_vertex =
            (edge0_vertex1 - edge0_vertex0) * u_up[i] / u_dw[i] + edge0_vertex0;
        Scalar edge1_vertex =
            (edge1_vertex1 - edge1_vertex0) * v_up[i] / v_dw[i] + edge1_vertex0;
        rst[i] = edge0_vertex - edge1_vertex;
    }
}

// ** this version can return the true x or y or z tolerance of the co-domain **
// eps is the interval [-eps,eps] we need to check
// if [-eps,eps] overlap, return true
// bbox_in_eps tell us if the box is totally in eps box
// ms is the minimum seperation
__device__ bool evaluate_bbox_one_dimension_vector_return_tolerance(
    Scalar *t_up,
    Scalar *t_dw,
    Scalar *u_up,
    Scalar *u_dw,
    Scalar *v_up,
    Scalar *v_dw,
    const Scalar *a0s,
    const Scalar *a1s,
    const Scalar *b0s,
    const Scalar *b1s,
    const Scalar *a0e,
    const Scalar *a1e,
    const Scalar *b0e,
    const Scalar *b1e,
    const int dimension,
    const bool check_vf,
    const Scalar eps,
    const Scalar ms,
    bool &bbox_in_eps,
    Scalar &tol)
{

    Scalar vs[8];
    int count = 0;
    bbox_in_eps = false;

    if (check_vf)
    {
        function_vf(
            a0s[dimension], a1s[dimension], b0s[dimension], b1s[dimension],
            a0e[dimension], a1e[dimension], b0e[dimension], b1e[dimension],
            t_up, t_dw, u_up, u_dw, v_up, v_dw, vs);
    }
    else
    {
        function_ee(
            a0s[dimension], a1s[dimension], b0s[dimension], b1s[dimension],
            a0e[dimension], a1e[dimension], b0e[dimension], b1e[dimension],
            t_up, t_dw, u_up, u_dw, v_up, v_dw, vs);
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


__device__ bool Origin_in_function_bounding_box_double_vector_return_tolerance(
    Singleinterval paras[3],
    const Scalar a0s[3],
    const Scalar a1s[3],
    const Scalar b0s[3],
    const Scalar b1s[3],
    const Scalar a0e[3],
    const Scalar a1e[3],
    const Scalar b0e[3],
    const Scalar b1e[3],
    const bool check_vf,
    const Scalar box[3],
    const Scalar ms,
    bool &box_in_eps,
    Scalar tolerance[3])
{

    box_in_eps = false;
    Scalar t_up[8];
    Scalar t_dw[8];
    Scalar u_up[8];
    Scalar u_dw[8];
    Scalar v_up[8];
    Scalar v_dw[8];
    Singleinterval *itv0 = &paras[0], *itv1 = &paras[1], *itv2 = &paras[2];

    convert_tuv_to_array(itv0, itv1, itv2, t_up, t_dw, u_up, u_dw, v_up, v_dw);

    bool ck;
    bool box_in[3];
    for (int i = 0; i < 3; i++)
    {

        ck = evaluate_bbox_one_dimension_vector_return_tolerance(
            t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s, a1s, b0s, b1s, a0e,
            a1e, b0e, b1e, i, check_vf, box[i], ms, box_in[i],
            tolerance[i]);

        if (!ck)
            return false;
    }
    if (box_in[0] && box_in[1] && box_in[2])
    {
        box_in_eps = true;
    }
    return true;
}
__device__ void bisect(const Singleinterval inter, interval_pair &out)
{
    Numccd low = inter.first;
    Numccd up = inter.second;

    // interval is [k1/pow(2,n1), k2/pow(2,n2)], k1,k2,n1,n2 are all not negative
    long k1 = low.first;
    int n1 = low.second;
    long k2 = up.first;
    int n2 = up.second;

    //assert(k1 >= 0 && k2 >= 0 && n1 >= 0 && n2 >= 0);

    //interval_pair out;
    long k;
    int n;
    int p;
    long r;
    if (n2 == n1)
    {
        p = reduction(k1 + k2, r);
        k = r;
        n = n2 - p + 1;
    }
    if (n2 > n1)
    {
        k = k1 * power(1, n2 - n1) + k2;
        // assert(k % 2 == 1);
        n = n2 + 1;
    }
    if (n2 < n1)
    {
        k = k1 + k2 * power(1, n1 - n2);
        // assert(k % 2 == 1);
        n = n1 + 1;
    }
    Numccd newnum(k, n);
    Singleinterval i1(low, newnum), i2(newnum, up);
    // std::cout<<"low,"<<Numccd2double(low)<<",up,"<<Numccd2double(up)<<", num, "<<Numccd2double(newnum)<<std::endl;
    // std::cout<<"new, k1, "<<newnum.first<<", n1, "<<newnum.second<<std::endl;
    // assert(
    //     Numccd2double(newnum) > Numccd2double(low)
    //     && Numccd2double(newnum) < Numccd2double(up));
    // interval_cp(i1,out.first);
    // interval_cp(i2,out.second);
    out.first = i1;
    out.second = i2;
}

__device__ bool RFclass::interval_root_finder_double_horizontal_tree(
    const Scalar tol[3],
    const Scalar co_domain_tolerance,
    const Singleinterval iset[3],
    const bool check_t_overlap,
    const Scalar
        max_t, // check interval [0, max_t] when check_t_overlap is set as TRUE
    Scalar &toi,
    const bool check_vf,
    const Scalar err[3],
    const Scalar ms,
    const Scalar a0s[3],
    const Scalar a1s[3],
    const Scalar b0s[3],
    const Scalar b1s[3],
    const Scalar a0e[3],
    const Scalar a1e[3],
    const Scalar b0e[3],
    const Scalar b1e[3],
    const int max_itr,
    Scalar &output_tolerance,
    int &overflow_flag)
{
    


    overflow_flag = NO_OVERFLOW;
    // if max_itr <0, output_tolerance= co_domain_tolerance;
    // else, output_tolearance will be the precision after iteration time > max_itr
    output_tolerance = co_domain_tolerance;

    // this is used to catch the tolerance for each level
    
    temp_output_tolerance = co_domain_tolerance;

    
    
    
    refine = 0;

    impact_ratio = 1;
    toi = SCALAR_LIMIT; //set toi as infinate
    // temp_toi is to catch the toi of each level

    temp_toi = toi;
    
    TOI.first = 4;
    TOI.second = 0; // set TOI as 4. this is to record the impact time of this level

    TOI_SKIP = TOI;               // this is to record the element that already small enough or contained in eps-box
    
    use_skip= false; // this is to record if TOI_SKIP is used.


    current_level = -2; // in the begining, current_level != level
    
    box_in_level = -2;  // this checks if all the boxes before this
    // level < tolerance. only true, we can return when we find one overlaps eps box and smaller than tolerance or eps-box
    
    this_level_less_tol = true;
     find_level_root= false;
    
     t_upper_bound= max_t; // 2*tol make it more conservative
    
    // Singleinterval test0=iset[0];
    // Singleinterval test1=iset[1];
    // Singleinterval test2=iset[2];
    //item init_item();
    
    
    istack.insertKey(item(iset, -1));
    
    while (!istack.empty())
    {
        
        if(overflow_flag>0){
            break;
        }
        
        
        current_item= istack.extractMin();
        
        current[0] = current_item.itv[0];
        current[1] = current_item.itv[1];
        current[2] = current_item.itv[2];
        level = current_item.level;
        
        // if this box is later than TOI_SKIP in time, we can skip this one.
        // TOI_SKIP is only updated when the box is small enough or totally contained in eps-box
        
        if (!less_than(current[0].first, TOI_SKIP))
        {
            continue;
        }
        if (box_in_level != level)
        { // before check a new level, set this_level_less_tol=true
            box_in_level = level;
            this_level_less_tol = true;
        }

        refine++;
        
        
        zero_in =
            Origin_in_function_bounding_box_double_vector_return_tolerance(
                current, a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, check_vf,
                err, ms, box_in, true_tol);
        if (!zero_in)
            continue;
        
        widths = width(current);

        tol_condition = true_tol[0] <= co_domain_tolerance && true_tol[1] <= co_domain_tolerance && true_tol[2] <= co_domain_tolerance;

        // Condition 1, stopping condition on t, u and v is satisfied. this is useless now since we have condition 2
       condition1 = widths.v[0] <= tol[0] && widths.v[1] <= tol[1] && widths.v[2] <= tol[2];

        // Condition 2, zero_in = true, box inside eps-box and in this level,
        // no box whose zero_in is true but box size larger than tolerance, can return
        condition2 = box_in && this_level_less_tol;
        if (!tol_condition)
        {
            this_level_less_tol = false;
            // this level has at least one box whose size > tolerance, thus we
            // cannot directly return if find one box whose size < tolerance or box-in
        }
        
        // Condition 3, in this level, we find a box that zero-in and size < tolerance.
        // and no other boxes whose zero-in is true in this level before this one is larger than tolerance, can return
        condition3 = this_level_less_tol;
        if (condition1 || condition2 || condition3)
        {
            TOI = current[0].first;
            // continue;
            toi = Numccd2double(TOI) * impact_ratio;
            //std::cout << "return 1" << std::endl;

            return true;
            // we don't need to compare with TOI_SKIP because we already continue
            // when t>=TOI_SKIP
        }

        // it used to be if(max_itr > 0). however, since we are limiting the heap size, truncation may happens because of max_itr, or heap size.
        if (1) 
        { // if max_itr < 0, then stop until stack empty
            if (current_level != level)
            {
                // output_tolerance=current_tolerance;
                // current_tolerance=0;
                current_level = level;
                find_level_root = false;
            }
            // current_tolerance=std::max(
            // std::max(std::max(current_tolerance,true_tol[0]),true_tol[1]),true_tol[2]
            // );
            if (!find_level_root)
            {
                TOI = current[0].first;
                // continue;
                temp_toi = Numccd2double(TOI) * impact_ratio;

                // if the real tolerance is larger than input, use the real one;
                // if the real tolerance is smaller than input, use input
                temp_output_tolerance = max(
                    max(
                        max(true_tol[0], true_tol[1]), true_tol[2]),
                    co_domain_tolerance);
                
                find_level_root =
                    true; // this ensures always find the earlist root
            }
            if (refine > max_itr)
            {
                overflow_flag=ITERATION_OVERFLOW;
                // std::cout<<"return from refine"<<std::endl;
                break;
            }
            // get the time of impact down here
        }

        // if this box is small enough, or inside of eps-box, then just continue,
        // but we need to record the collision time
        if (tol_condition || box_in)
        {
            if (less_than(current[0].first, TOI_SKIP))
            {
                TOI_SKIP = current[0].first;
            }
            use_skip = true;
            continue;
        }
        
        

        check[0] = false;
        check[1] = false;
        check[2] = false;
        for (int i = 0; i < 3; i++)
        {
            widthratio.v[i] = widths.v[i] / tol[i];
            if (widths.v[i] > tol[i])
                check[i] = true; // means this need to be checked
        }

        split_i = -1;
        
        for (int i = 0; i < 3; i++)
        {
            if (check[i])
            {
                if (check[(i + 1) % 3] && check[(i + 2) % 3])
                {
                    if (widthratio.v[i] >= widthratio.v[(i + 1) % 3] && widthratio.v[i] >= widthratio.v[(i + 2) % 3])
                    {
                        split_i = i;
                        break;
                    }
                }
                if (check[(i + 1) % 3] && !check[(i + 2) % 3])
                {
                    if (widthratio.v[i] >= widthratio.v[(i + 1) % 3])
                    {
                        split_i = i;
                        break;
                    }
                }
                if (!check[(i + 1) % 3] && check[(i + 2) % 3])
                {
                    if (widthratio.v[i] >= widthratio.v[(i + 2) % 3])
                    {
                        split_i = i;
                        break;
                    }
                }
                if (!check[(i + 1) % 3] && !check[(i + 2) % 3])
                {

                    split_i = i;
                    break;
                }
            }
        }
  
        
        bisect(current[split_i], halves);
        if (!less_than(halves.first.first, halves.first.second))
        {
            // std::cout << "OVERFLOW HAPPENS WHEN SPLITTING INTERVALS"
            //           << std::endl;
            overflow_flag = BISECTION_OVERFLOW;
            break;
        }
        if (!less_than(halves.second.first, halves.second.second))
        {
            // std::cout << "OVERFLOW HAPPENS WHEN SPLITTING INTERVALS"
            //           << std::endl;
            overflow_flag = BISECTION_OVERFLOW;
            break;
        }
        if (check_vf)
        {
            //std::cout<<"*** check_vf"<<std::endl;
            if (split_i == 1)
            {
                // assert(sum_no_larger_1(halves.first.first, current[2].first)==sum_no_larger_1_Rational(halves.first.first, current[2].first));
                // assert(sum_no_larger_1(halves.second.first, current[2].first)==sum_no_larger_1_Rational(halves.second.first, current[2].first));

                if (sum_no_larger_1(halves.second.first, current[2].first))
                {

                    current[split_i] = halves.second;
                    inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = HEAP_OVERFLOW;
                    }
                }
                if (sum_no_larger_1(halves.first.first, current[2].first))
                {
                    current[split_i] = halves.first;
                    inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = HEAP_OVERFLOW;
                    }
                }
            }

            if (split_i == 2)
            {
                //assert(sum_no_larger_1(halves.first.first, current[1].first)==sum_no_larger_1_Rational(halves.first.first, current[1].first));
                //assert(sum_no_larger_1(halves.second.first, current[1].first)==sum_no_larger_1_Rational(halves.second.first, current[1].first));

                if (sum_no_larger_1(halves.second.first, current[1].first))
                {
                    current[split_i] = halves.second;
                    inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = HEAP_OVERFLOW;
                    }
                }
                if (sum_no_larger_1(halves.first.first, current[1].first))
                {
                    current[split_i] = halves.first;
                    inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = HEAP_OVERFLOW;
                    }
                }
            }
            if (split_i == 0)
            {
                if (check_t_overlap)
                {
                    if (interval_overlap_region(
                            halves.second, 0, t_upper_bound))
                    {
                        current[split_i] = halves.second;
                        inserted = istack.insertKey(item(current, level + 1));
                        if (inserted == false)
                        {
                            overflow_flag = HEAP_OVERFLOW;
                        }
                    }
                    if (interval_overlap_region(
                            halves.first, 0, t_upper_bound))
                    {
                        current[split_i] = halves.first;
                        inserted = istack.insertKey(item(current, level + 1));
                        if (inserted == false)
                        {
                            overflow_flag = HEAP_OVERFLOW;
                        }
                    }
                }
                else
                {
                    current[split_i] = halves.second;
                    inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = HEAP_OVERFLOW;
                    }
                    current[split_i] = halves.first;
                    inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = HEAP_OVERFLOW;
                    }
                }
            }
        }
        else
        {
            if (check_t_overlap && split_i == 0)
            {
                if (interval_overlap_region(
                        halves.second, 0, t_upper_bound))
                {
                    current[split_i] = halves.second;
                    inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = HEAP_OVERFLOW;
                    }
                }
                if (interval_overlap_region(halves.first, 0, t_upper_bound))
                {
                    current[split_i] = halves.first;
                    inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = HEAP_OVERFLOW;
                    }
                }
            }
            else
            {
                current[split_i] = halves.second;
                inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
                current[split_i] = halves.first;
                inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
        }
    }
    
    if (overflow_flag > 0)
    {
        toi = temp_toi;
        output_tolerance = temp_output_tolerance;


        return true;
    }
    
    if (use_skip)
    {
        toi = Numccd2double(TOI_SKIP) * impact_ratio;
        return true;
    }

    
    return false;
}

__device__ bool interval_root_finder_double_horizontal_tree(
    RFvar & var,
    const Scalar tol[3],
    const Scalar co_domain_tolerance,
    Scalar &toi,
    const bool check_vf,
    const Scalar err[3], // this is the maximum error on each axis when calculating the vertices, err, aka, filter
    const Scalar ms,
    const Scalar a0s[3],
    const Scalar a1s[3],
    const Scalar b0s[3],
    const Scalar b1s[3],
    const Scalar a0e[3],
    const Scalar a1e[3],
    const Scalar b0e[3],
    const Scalar b1e[3],
    const Scalar max_time,
    const int max_itr,
    Scalar &output_tolerance,
    int &overflow_flag)
{
    return true;
    var.check_t_overlap =
        max_time == 1
            ? false
            : true; // if input max_time = 1, then no need to check overlap

    
    var.low_number.first = 0;
    var.low_number.second = 0; // low_number=0;
    
    var.up_number.first = 1;
    var.up_number.second = 0; // up_number=1;
    // initial interval [0,1]
    
    var.init_interval.first = var.low_number;
    var.init_interval.second = var.up_number;
    //build interval set [0,1]x[0,1]x[0,1]
    
    var.iset[0] = var.init_interval;
    var.iset[1] = var.init_interval;
    var.iset[2] = var.init_interval;
    
    var.result = var.rf.interval_root_finder_double_horizontal_tree(
        tol, co_domain_tolerance, var.iset, var.check_t_overlap, max_time, toi,
        check_vf, err, ms, a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, max_itr,
        output_tolerance, overflow_flag);

    return var.result;
}

__device__ Scalar max_linf_dist(const VectorMax3d &p1, const VectorMax3d &p2, Scalar& r, int &i)
{
    r = 0;
    for (i = 0; i < 3; i++)
    {
        
        if (r < fabs(p1.v[i] - p2.v[i]))
        {
            r = fabs(p1.v[i] - p2.v[i]);
        }
    }
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
    const VectorMax3d &p4e, Scalar&r, Scalar &r1, Scalar &temp, int &i)
{
    r = 0;
    temp = 0;
    temp = max_linf_dist(p1e, p1, r1, i);
    if (r < temp)
        r = temp;
    temp = max_linf_dist(p2e, p2, r1, i);
    if (r < temp)
        r = temp;
    temp = max_linf_dist(p3e, p3, r1, i);
    if (r < temp)
        r = temp;
    temp = max_linf_dist(p4e, p4, r1, i);
    if (r < temp)
        r = temp;
    return r;
}
__device__ VectorMax3d compute_face_vertex_tolerance_3d_new(
    ERRvar& var, const CCDdata *data_in,
    const Scalar tolerance)
{
    
    var.v0s.v[0]=data_in->v0s[0]; var.v0s.v[1]=data_in->v0s[1]; var.v0s.v[2]=data_in->v0s[2];
    var.v0s.v[0]=data_in->v1s[0]; var.v0s.v[1]=data_in->v1s[1]; var.v0s.v[2]=data_in->v1s[2];
    var.v0s.v[0]=data_in->v2s[0]; var.v0s.v[1]=data_in->v2s[1]; var.v0s.v[2]=data_in->v2s[2];
    var.v0s.v[0]=data_in->v3s[0]; var.v0s.v[1]=data_in->v3s[1]; var.v0s.v[2]=data_in->v3s[2];
    var.v0s.v[0]=data_in->v0e[0]; var.v0s.v[1]=data_in->v0e[1]; var.v0s.v[2]=data_in->v0e[2];
    var.v0s.v[0]=data_in->v1e[0]; var.v0s.v[1]=data_in->v1e[1]; var.v0s.v[2]=data_in->v1e[2];
    var.v0s.v[0]=data_in->v2e[0]; var.v0s.v[1]=data_in->v2e[1]; var.v0s.v[2]=data_in->v2e[2];
    var.v0s.v[0]=data_in->v3e[0]; var.v0s.v[1]=data_in->v3e[1]; var.v0s.v[2]=data_in->v3e[2];
    
    VecMinus(var.v0s, var.v1s,var.p000); 
    VecMinus(var.v0s , var.v3s,var.p001);
    VecSum(var.v2s,var.v3s,var.p011);
    VecMinus(var.p011,var.v1s,var.p011);
    VecMinus(var.v0s,var.p011,var.p011);
    VecMinus(var.v0s , var.v2s, var.p010);
    VecMinus(var.v0e , var.v1e, var.p100);
    VecMinus(var.v0e , var.v3e, var.p101);
    VecSum(var.v2e , var.v3e, var.p111);
    VecMinus(var.p111 , var.v1e, var.p111);
    VecMinus(var.v0e , var.p111, var.p111);
    VecMinus(var.v0e , var.v2e, var.p110);
    //
    

    var.dl = 0;
    var.edge0_length = 0;
    var.edge1_length = 0;
    
    var.dl = 3 * max_linf_4(var.p000, var.p001, var.p011, var.p010, var.p100, var.p101, var.p111, var.p110, 
    var.m_r,var.m_r1,var.m_temp,var.m_i);
    var.edge0_length =
        3 * max_linf_4(var.p000, var.p100, var.p101, var.p001, var.p010, var.p110, var.p111, var.p011,
        var.m_r,var.m_r1,var.m_temp,var.m_i);
    var.edge1_length =
        3 * max_linf_4(var.p000, var.p100, var.p110, var.p010, var.p001, var.p101, var.p111, var.p011,
        var.m_r,var.m_r1,var.m_temp,var.m_i);
    
    var.res.v[0]=tolerance / var.dl;
    var.res.v[1]=tolerance / var.edge0_length;
    var.res.v[2]=tolerance / var.edge1_length;
    return var.res;
}
__device__ VectorMax3d compute_edge_edge_tolerance_new(
    ERRvar& var, const CCDdata *data_in,
    const Scalar tolerance)
{
    //return VectorMax3d(0,0,0);
    var.v0s.v[0]=data_in->v0s[0]; var.v0s.v[1]=data_in->v0s[1]; var.v0s.v[2]=data_in->v0s[2];
    var.v0s.v[0]=data_in->v1s[0]; var.v0s.v[1]=data_in->v1s[1]; var.v0s.v[2]=data_in->v1s[2];
    var.v0s.v[0]=data_in->v2s[0]; var.v0s.v[1]=data_in->v2s[1]; var.v0s.v[2]=data_in->v2s[2];
    var.v0s.v[0]=data_in->v3s[0]; var.v0s.v[1]=data_in->v3s[1]; var.v0s.v[2]=data_in->v3s[2];
    var.v0s.v[0]=data_in->v0e[0]; var.v0s.v[1]=data_in->v0e[1]; var.v0s.v[2]=data_in->v0e[2];
    var.v0s.v[0]=data_in->v1e[0]; var.v0s.v[1]=data_in->v1e[1]; var.v0s.v[2]=data_in->v1e[2];
    var.v0s.v[0]=data_in->v2e[0]; var.v0s.v[1]=data_in->v2e[1]; var.v0s.v[2]=data_in->v2e[2];
    var.v0s.v[0]=data_in->v3e[0]; var.v0s.v[1]=data_in->v3e[1]; var.v0s.v[2]=data_in->v3e[2];
    VecMinus(var.v0s , var.v2s, var.p000);
    VecMinus(var.v0s , var.v3s, var.p001);
    VecMinus(var.v1s , var.v3s, var.p011);
    VecMinus(var.v1s , var.v2s, var.p010);
    VecMinus(var.v0e , var.v2e, var.p100);
    VecMinus(var.v0e , var.v3e, var.p101);
    VecMinus(var.v1e , var.v3e, var.p111);
    VecMinus(var.v1e , var.v2e, var.p110);
    // var.p000 = var.v0s - var.v2s;
    // var.p001 = var.v0s - var.v3s;
    // var.p011 = var.v1s - var.v3s;
    // var.p010 = var.v1s - var.v2s;
    // var.p100 = var.v0e - var.v2e;
    // var.p101 = var.v0e - var.v3e;
    // var.p111 = var.v1e - var.v3e;
    // var.p110 = var.v1e - var.v2e;
    var.dl = 0;
    var.edge0_length = 0;
    var.edge1_length = 0;

    var.dl = 3 * max_linf_4(var.p000, var.p001, var.p011, var.p010, var.p100, var.p101, var.p111, var.p110, 
    var.m_r,var.m_r1,var.m_temp,var.m_i);
    var.edge0_length =
        3 * max_linf_4(var.p000, var.p100, var.p101, var.p001, var.p010, var.p110, var.p111, var.p011, 
        var.m_r,var.m_r1,var.m_temp,var.m_i);
    var.edge1_length =
        3 * max_linf_4(var.p000, var.p100, var.p110, var.p010, var.p001, var.p101, var.p111, var.p011, 
        var.m_r,var.m_r1,var.m_temp,var.m_i);
    var.res.v[0]=tolerance / var.dl;
    var.res.v[1]=tolerance / var.edge0_length;
    var.res.v[2]=tolerance / var.edge1_length;
    return var.res;
}

__device__ __host__ void get_numerical_error(
    ERRvar &var,
    const VectorMax3d vertices[8], 
    const bool &check_vf,
    const bool using_minimum_separation,
    Scalar error[3])
{
    
    
    if (!using_minimum_separation)
    {
#ifdef GPUTI_USE_DOUBLE_PRECISION
        var.eefilter = 6.217248937900877e-15;
        var.vffilter = 6.661338147750939e-15;
#else
        var.eefilter = 3.337861e-06;
        var.vffilter = 3.576279e-06;
#endif
    }
    else // using minimum separation
    {
#ifdef GPUTI_USE_DOUBLE_PRECISION
        var.eefilter = 7.105427357601002e-15;
        var.vffilter = 7.549516567451064e-15;
#else
        var.eefilter = 3.814698e-06;
        var.vffilter = 4.053116e-06;
#endif
    }
    
    fabs(vertices[0].v[0],var.xmax);
    fabs(vertices[0].v[1],var.ymax);
    fabs(vertices[0].v[2],var.zmax);
    
    // find the biggest
    fabs(vertices[0].v[0],var.dl);if (var.xmax < var.dl){var.xmax = var.dl;}//
    fabs(vertices[0].v[1],var.dl);if (var.ymax < var.dl){var.ymax = var.dl;}
    fabs(vertices[0].v[2],var.dl);if (var.zmax < var.dl){var.zmax = var.dl;}
    fabs(vertices[1].v[0],var.dl);if (var.xmax < var.dl){var.xmax = var.dl;}//
    fabs(vertices[1].v[1],var.dl);if (var.ymax < var.dl){var.ymax = var.dl;}
    fabs(vertices[1].v[2],var.dl);if (var.zmax < var.dl){var.zmax = var.dl;}
    fabs(vertices[2].v[0],var.dl);if (var.xmax < var.dl){var.xmax = var.dl;}//
    fabs(vertices[2].v[1],var.dl);if (var.ymax < var.dl){var.ymax = var.dl;}
    fabs(vertices[2].v[2],var.dl);if (var.zmax < var.dl){var.zmax = var.dl;}
    fabs(vertices[3].v[0],var.dl);if (var.xmax < var.dl){var.xmax = var.dl;}//
    fabs(vertices[3].v[1],var.dl);if (var.ymax < var.dl){var.ymax = var.dl;}
    fabs(vertices[3].v[2],var.dl);if (var.zmax < var.dl){var.zmax = var.dl;}
    fabs(vertices[4].v[0],var.dl);if (var.xmax < var.dl){var.xmax = var.dl;}//
    fabs(vertices[4].v[1],var.dl);if (var.ymax < var.dl){var.ymax = var.dl;}
    fabs(vertices[4].v[2],var.dl);if (var.zmax < var.dl){var.zmax = var.dl;}
    fabs(vertices[5].v[0],var.dl);if (var.xmax < var.dl){var.xmax = var.dl;}//
    fabs(vertices[5].v[1],var.dl);if (var.ymax < var.dl){var.ymax = var.dl;}
    fabs(vertices[5].v[2],var.dl);if (var.zmax < var.dl){var.zmax = var.dl;}
    fabs(vertices[6].v[0],var.dl);if (var.xmax < var.dl){var.xmax = var.dl;}//
    fabs(vertices[6].v[1],var.dl);if (var.ymax < var.dl){var.ymax = var.dl;}
    fabs(vertices[6].v[2],var.dl);if (var.zmax < var.dl){var.zmax = var.dl;}
    fabs(vertices[7].v[0],var.dl);if (var.xmax < var.dl){var.xmax = var.dl;}//
    fabs(vertices[7].v[1],var.dl);if (var.ymax < var.dl){var.ymax = var.dl;}
    fabs(vertices[7].v[2],var.dl);if (var.zmax < var.dl){var.zmax = var.dl;}    
    
    var.delta_x = var.xmax > 1 ? var.xmax : 1;
    var.delta_y = var.ymax > 1 ? var.ymax : 1;
    var.delta_z = var.zmax > 1 ? var.zmax : 1;
    if (!check_vf)
    {
        error[0] = var.delta_x * var.delta_x * var.delta_x * var.eefilter;
        error[1] = var.delta_y * var.delta_y * var.delta_y * var.eefilter;
        error[2] = var.delta_z * var.delta_z * var.delta_z * var.eefilter;
    }
    else
    {
        error[0] = var.delta_x * var.delta_x * var.delta_x * var.vffilter;
        error[1] = var.delta_y * var.delta_y * var.delta_y * var.vffilter;
        error[2] = var.delta_z * var.delta_z * var.delta_z * var.vffilter;
    }
    return;
}

__device__ bool CCD_Solver(
    CCDvar &vars,
    CCDdata *data_in,
    const Scalar *err,
    const Scalar ms,
    Scalar &toi,
    Scalar tolerance,
    Scalar t_max,
    const int max_itr,
    Scalar &output_tolerance,
    bool no_zero_toi,
    int &overflow_flag,
    bool is_vf)
{
    
    overflow_flag = 0;
    
    
   
    vars.no_zero_toi_iter = 0;
    
    vars.tolerance_in = tolerance;
    vars.ms_in = ms;
    
    do
    {
        if(is_vf){
           vars.tol_v= compute_face_vertex_tolerance_3d_new(vars.errvar, data_in, vars.tolerance_in);
        }
        else{
            
            vars.tol_v=
            compute_edge_edge_tolerance_new(vars.errvar, data_in, vars.tolerance_in);
        }
        
        
        vars.tol[0] = vars.tol_v.v[0];
        vars.tol[1] = vars.tol_v.v[1];
        vars.tol[2] = vars.tol_v.v[2];
        //////////////////////////////////////////////////////////
        
        if(err[0] < 0){
            for (vars.itr = 0; vars.itr < 3; vars.itr++)
            {
                vars.vlist[0].v[vars.itr] = data_in->v0s[vars.itr];
                vars.vlist[1].v[vars.itr] = data_in->v1s[vars.itr];
                vars.vlist[2].v[vars.itr] = data_in->v2s[vars.itr];
                vars.vlist[3].v[vars.itr] = data_in->v3s[vars.itr];
                vars.vlist[4].v[vars.itr] = data_in->v0e[vars.itr];
                vars.vlist[5].v[vars.itr] = data_in->v1e[vars.itr];
                vars.vlist[6].v[vars.itr] = data_in->v2e[vars.itr];
                vars.vlist[7].v[vars.itr] = data_in->v3e[vars.itr];
            }
            vars.use_ms = ms > 0;
            get_numerical_error(vars.errvar, vars.vlist, is_vf, vars.use_ms, vars.err1);
        }
        else{
            vars.err1[0] = err[0];
            vars.err1[1] = err[1];
            vars.err1[2] = err[2];
        }
        
        //////////////////////////////////////////////////////////
        //return true;
        vars.tmp_is_impacting = interval_root_finder_double_horizontal_tree(
            vars.rfvar,
            vars.tol, vars.tolerance_in, toi, is_vf, vars.err1, vars.ms_in, data_in->v0s,
            data_in->v1s, data_in->v2s, data_in->v3s,
            data_in->v0e, data_in->v1e, data_in->v2e,
            data_in->v3e, t_max, max_itr, output_tolerance, overflow_flag);
        
        
        if (overflow_flag)
        {

            return true;
        }
        if (vars.no_zero_toi_iter == 0)
        {
            // This will be the final output because we might need to
            // perform CCD again if the toi is zero. In which case we will
            // use a smaller t_max for more time resolution.
            vars.is_impacting = vars.tmp_is_impacting;
        }
        else
        {
            toi = vars.tmp_is_impacting ? toi : t_max;
        }

        // This modification is for CCD-filtered line-search (e.g., IPC)
        // strategies for dealing with toi = 0:
        // 1. shrink t_max (when reaches max_itr),
        // 2. shrink tolerance (when not reach max_itr and tolerance is big) or
        // ms (when tolerance is too small comparing with ms)
        if (vars.tmp_is_impacting && toi == 0 && no_zero_toi)
        {

            // meaning reaches max_itr, need to shrink the t_max to return a more accurate result to reach target tolerance.
            if (output_tolerance > vars.tolerance_in)
            {
                t_max *= 0.9;
            }
            else
            { // meaning the given tolerance or ms is too large. need to shrink them,
                if (10 * vars.tolerance_in < vars.ms_in)
                { // ms is too large, shrink it
                    vars.ms_in *= 0.5;
                }
                else
                { // tolerance is too large, shrink it
                    vars.tolerance_in *= 0.5;
                }
            }
        }

        vars.no_zero_toi_iter++;

        // Only perform a second iteration if toi == 0.
        // WARNING: This option assumes the initial distance is not zero.
    } while (no_zero_toi && vars.no_zero_toi_iter < vars.MAX_NO_ZERO_TOI_ITER && vars.tmp_is_impacting && toi == 0);
    return vars.is_impacting;
}

__device__ bool vertexFaceCCD_double(
    CCDvar &vars,
    CCDdata *data_in,
    const Scalar *err,
    const Scalar ms,
    Scalar &toi,
    Scalar tolerance,
    Scalar t_max,
    const int max_itr,
    Scalar &output_tolerance,
    bool no_zero_toi,
    int &overflow_flag)
{
    
    vars.res = CCD_Solver(vars, data_in, err, ms, toi, tolerance, t_max, max_itr, output_tolerance, no_zero_toi, overflow_flag, true);
    return vars.res;
}

__device__ bool edgeEdgeCCD_double(
    CCDvar &vars,
    CCDdata *data_in,
    const Scalar *err,
    const Scalar ms,
    Scalar &toi,
    Scalar tolerance,
    Scalar t_max,
    const int max_itr,
    Scalar &output_tolerance,
    bool no_zero_toi,
    int &overflow_flag)
{
    
    vars.res = CCD_Solver(vars, data_in, err, ms, toi, tolerance, t_max, max_itr, output_tolerance, no_zero_toi, overflow_flag, false);
    return vars.res;
}