#include <gputi/root_finder.h>
#include <gputi/queue.h>
#include <iostream>
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
__device__ __host__ VectorMax3d::VectorMax3d(Scalar a, Scalar b, Scalar c)
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

    Scalar *vs = new Scalar[8];
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
    delete[] vs;
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
    Singleinterval *paras,
    const Scalar *a0s,
    const Scalar *a1s,
    const Scalar *b0s,
    const Scalar *b1s,
    const Scalar *a0e,
    const Scalar *a1e,
    const Scalar *b0e,
    const Scalar *b1e,
    const bool check_vf,
    const Scalar *box,
    const Scalar ms,
    bool &box_in_eps,
    Scalar *tolerance)
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

__device__ void bisect(const Singleinterval *inter, interval_pair &out)
{
    Numccd low = inter->first;
    Numccd up = inter->second;

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

__device__ bool interval_root_finder_double_horizontal_tree(
    const Scalar *tol,
    const Scalar co_domain_tolerance,
    const Singleinterval *iset,
    const bool check_t_overlap,
    const Scalar
        max_t, // check interval [0, max_t] when check_t_overlap is set as TRUE
    Scalar &toi,
    const bool check_vf,
    const Scalar *err,
    const Scalar ms,
    const Scalar *a0s,
    const Scalar *a1s,
    const Scalar *b0s,
    const Scalar *b1s,
    const Scalar *a0e,
    const Scalar *a1e,
    const Scalar *b0e,
    const Scalar *b1e,
    const int max_itr,
    Scalar &output_tolerance,
    int &overflow_flag)
{
    
    //return true;
    overflow_flag = 0;
    // if max_itr <0, output_tolerance= co_domain_tolerance;
    // else, output_tolearancewill be the precision after iteration time > max_itr
    output_tolerance = co_domain_tolerance;

    // this is used to catch the tolerance for each level
    Scalar temp_output_tolerance = co_domain_tolerance;

    
    
    // current intervals
    Singleinterval *current = new Singleinterval[3];
    Scalar *true_tol = new Scalar[3];
    int refine = 0;
    Scalar impact_ratio = 1;

    toi = SCALAR_LIMIT; //set toi as infinate
    // temp_toi is to catch the toi of each level
    Scalar temp_toi = toi;
    Numccd TOI;
    TOI.first = 4;
    TOI.second = 0; // set TOI as 4. this is to record the impact time of this level
    Numccd TOI_SKIP =
        TOI;               // this is to record the element that already small enough or contained in eps-box
    bool use_skip = false; // this is to record if TOI_SKIP is used.

    int rnbr = 0;
    int current_level = -2; // in the begining, current_level != level
    int box_in_level = -2;  // this checks if all the boxes before this
    // level < tolerance. only true, we can return when we find one overlaps eps box and smaller than tolerance or eps-box
    bool this_level_less_tol = true;
    bool find_level_root = false;
    // Scalar current_tolerance=std::numeric_limits<Scalar>::infinity(); // set returned tolerance as infinite
    Scalar t_upper_bound = max_t; // 2*tol make it more conservative
    
    MinHeap istack;
    
    // Singleinterval test0=iset[0];
    // Singleinterval test1=iset[1];
    // Singleinterval test2=iset[2];
    //item init_item();
    

    istack.insertKey(item(iset, -1));
    
    while (!istack.empty())
    {
        
        item current_item;
        
        current_item= istack.extractMin();
        
        current[0] = current_item.itv[0];
        current[1] = current_item.itv[1];
        current[2] = current_item.itv[2];
        int level = current_item.level;
        
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
        bool zero_in;
        bool box_in;
        
        zero_in =
            Origin_in_function_bounding_box_double_vector_return_tolerance(
                current, a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, check_vf,
                err, ms, box_in, true_tol);
        if (!zero_in)
            continue;
        
        VectorMax3d widths = width(current);

        bool tol_condition = true_tol[0] <= co_domain_tolerance && true_tol[1] <= co_domain_tolerance && true_tol[2] <= co_domain_tolerance;

        // Condition 1, stopping condition on t, u and v is satisfied. this is useless now since we have condition 2
        bool condition1 = widths.v[0] <= tol[0] && widths.v[1] <= tol[1] && widths.v[2] <= tol[2];

        // Condition 2, zero_in = true, box inside eps-box and in this level,
        // no box whose zero_in is true but box size larger than tolerance, can return
        bool condition2 = box_in && this_level_less_tol;
        if (!tol_condition)
        {
            this_level_less_tol = false;
            // this level has at least one box whose size > tolerance, thus we
            // cannot directly return if find one box whose size < tolerance or box-in
        }
        
        // Condition 3, in this level, we find a box that zero-in and size < tolerance.
        // and no other boxes whose zero-in is true in this level before this one is larger than tolerance, can return
        bool condition3 = this_level_less_tol;
        if (condition1 || condition2 || condition3)
        {
            TOI = current[0].first;
            rnbr++;
            // continue;
            toi = Numccd2double(TOI) * impact_ratio;
            //std::cout << "return 1" << std::endl;
            delete[] current;
            delete[] true_tol;
            return true;
            // we don't need to compare with TOI_SKIP because we already continue
            // when t>=TOI_SKIP
        }

        if (max_itr > 0)
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

                rnbr++;
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
                toi = temp_toi;
                output_tolerance = temp_output_tolerance;

                // std::cout<<"return from refine"<<std::endl;
                delete[] true_tol;
                delete[] current;
                return true;
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
        
        bool check[3];
        VectorMax3d widthratio;

        check[0] = false;
        check[1] = false;
        check[2] = false;
        for (int i = 0; i < 3; i++)
        {
            widthratio.v[i] = widths.v[i] / tol[i];
            if (widths.v[i] > tol[i])
                check[i] = true; // means this need to be checked
        }

        int split_i = -1;
        
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
        // if (split_i < 0)
        // {
        //     // std::cout
        //     //     << "ERROR OCCURRED HERE, DID NOT FIND THE RIGHT DIMENSION TO SPLIT"
        //         // << std::endl;
        // }
        // Bisect the next dimension that is greater than its tolerance
        // int split_i;
        // for (int i = 1; i <= 3; i++) {
        //     split_i = (last_split + i) % 3;
        //     if (widths(split_i) > tol(split_i)) {
        //         break;
        //     }
        // }
        
        interval_pair halves;
        Singleinterval *bisect_inter = &(current[split_i]);
        
        bisect(bisect_inter, halves);
        if (!less_than(halves.first.first, halves.first.second))
        {
            // std::cout << "OVERFLOW HAPPENS WHEN SPLITTING INTERVALS"
            //           << std::endl;
            overflow_flag = 1;
            delete[] current;
            delete[] true_tol;
            return true;
        }
        if (!less_than(halves.second.first, halves.second.second))
        {
            // std::cout << "OVERFLOW HAPPENS WHEN SPLITTING INTERVALS"
            //           << std::endl;
            overflow_flag = 1;
            delete[] true_tol;
            delete[] current;
            return true;
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
                    bool inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = 2;
                    }
                }
                if (sum_no_larger_1(halves.first.first, current[2].first))
                {
                    current[split_i] = halves.first;
                    bool inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = 2;
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
                    bool inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = 2;
                    }
                }
                if (sum_no_larger_1(halves.first.first, current[1].first))
                {
                    current[split_i] = halves.first;
                    bool inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = 2;
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
                        bool inserted = istack.insertKey(item(current, level + 1));
                        if (inserted == false)
                        {
                            overflow_flag = 2;
                        }
                    }
                    if (interval_overlap_region(
                            halves.first, 0, t_upper_bound))
                    {
                        current[split_i] = halves.first;
                        bool inserted = istack.insertKey(item(current, level + 1));
                        if (inserted == false)
                        {
                            overflow_flag = 2;
                        }
                    }
                }
                else
                {
                    current[split_i] = halves.second;
                    bool inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = 2;
                    }
                    current[split_i] = halves.first;
                    inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = 2;
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
                    bool inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = 2;
                    }
                }
                if (interval_overlap_region(halves.first, 0, t_upper_bound))
                {
                    current[split_i] = halves.first;
                    bool inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = 2;
                    }
                }
            }
            else
            {
                current[split_i] = halves.second;
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = 2;
                }
                current[split_i] = halves.first;
                inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = 2;
                }
            }
        }
    }
    
    if (overflow_flag > 0)
    {
        delete[] current;
        delete[] true_tol;
        return true;
    }
    
    if (use_skip)
    {
        toi = Numccd2double(TOI_SKIP) * impact_ratio;
        delete[] current;
        delete[] true_tol;
        return true;
    }
    //return true;
    delete[] current;
    //return true;
    delete[] true_tol;
    
    return false;
}

__device__ bool interval_root_finder_double_horizontal_tree(
    const Scalar *tol,
    const Scalar co_domain_tolerance,
    Scalar &toi,
    const bool check_vf,
    const Scalar *err, // this is the maximum error on each axis when calculating the vertices, err, aka, filter
    const Scalar ms,
    const Scalar *a0s,
    const Scalar *a1s,
    const Scalar *b0s,
    const Scalar *b1s,
    const Scalar *a0e,
    const Scalar *a1e,
    const Scalar *b0e,
    const Scalar *b1e,
    const Scalar max_time,
    const int max_itr,
    Scalar &output_tolerance,
    int &overflow_flag)
{

    bool check_t_overlap =
        max_time == 1
            ? false
            : true; // if input max_time = 1, then no need to check overlap

    Numccd low_number;
    low_number.first = 0;
    low_number.second = 0; // low_number=0;
    Numccd up_number;
    up_number.first = 1;
    up_number.second = 0; // up_number=1;
    // initial interval [0,1]
    Singleinterval init_interval;
    init_interval.first = low_number;
    init_interval.second = up_number;
    //build interval set [0,1]x[0,1]x[0,1]
    Singleinterval *iset = new Singleinterval[3];
    iset[0] = init_interval;
    iset[1] = init_interval;
    iset[2] = init_interval;
    
    bool result = interval_root_finder_double_horizontal_tree(
        tol, co_domain_tolerance, iset, check_t_overlap, max_time, toi,
        check_vf, err, ms, a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, max_itr,
        output_tolerance, overflow_flag);
    //return true;
    delete[] iset;
    return result;
}

__device__ Scalar max_linf_dist(const VectorMax3d &p1, const VectorMax3d &p2)
{
    Scalar r = 0;
    for (int i = 0; i < 3; i++)
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
    const VectorMax3d &p4e)
{
    Scalar r = 0, temp = 0;
    temp = max_linf_dist(p1e, p1);
    if (r < temp)
        r = temp;
    temp = max_linf_dist(p2e, p2);
    if (r < temp)
        r = temp;
    temp = max_linf_dist(p3e, p3);
    if (r < temp)
        r = temp;
    temp = max_linf_dist(p4e, p4);
    if (r < temp)
        r = temp;
    return r;
}
__device__ VectorMax3d compute_face_vertex_tolerance_3d_new(
    const CCDdata *data_in,
    const Scalar tolerance)
{
    VectorMax3d vs(data_in->v0s[0], data_in->v0s[1], data_in->v0s[2]);
    VectorMax3d f0s(data_in->v1s[0], data_in->v1s[1], data_in->v1s[2]);
    VectorMax3d f1s(data_in->v2s[0], data_in->v2s[1], data_in->v2s[2]);
    VectorMax3d f2s(data_in->v3s[0], data_in->v3s[1], data_in->v3s[2]);
    VectorMax3d ve(data_in->v0e[0], data_in->v0e[1], data_in->v0e[2]);
    VectorMax3d f0e(data_in->v1e[0], data_in->v1e[1], data_in->v1e[2]);
    VectorMax3d f1e(data_in->v2e[0], data_in->v2e[1], data_in->v2e[2]);
    VectorMax3d f2e(data_in->v3e[0], data_in->v3e[1], data_in->v3e[2]);
    VectorMax3d p000 = vs - f0s, p001 = vs - f2s,
                p011 = vs - (f1s + f2s - f0s), p010 = vs - f1s;
    VectorMax3d p100 = ve - f0e, p101 = ve - f2e,
                p111 = ve - (f1e + f2e - f0e), p110 = ve - f1e;
    Scalar dl = 0;
    Scalar edge0_length = 0;
    Scalar edge1_length = 0;
    dl = 3 * max_linf_4(p000, p001, p011, p010, p100, p101, p111, p110);
    edge0_length =
        3 * max_linf_4(p000, p100, p101, p001, p010, p110, p111, p011);
    edge1_length =
        3 * max_linf_4(p000, p100, p110, p010, p001, p101, p111, p011);
    return VectorMax3d(
        tolerance / dl, tolerance / edge0_length, tolerance / edge1_length);
}
__device__ VectorMax3d compute_edge_edge_tolerance_new(
    const CCDdata *data_in,
    const Scalar tolerance)
{
    VectorMax3d edge0_vertex0_start(data_in->v0s[0], data_in->v0s[1], data_in->v0s[2]); // a0s
    VectorMax3d edge0_vertex1_start(data_in->v1s[0], data_in->v1s[1], data_in->v1s[2]); // a1s
    VectorMax3d edge1_vertex0_start(data_in->v2s[0], data_in->v2s[1], data_in->v2s[2]); // b0s
    VectorMax3d edge1_vertex1_start(data_in->v3s[0], data_in->v3s[1], data_in->v3s[2]); // b1s
    VectorMax3d edge0_vertex0_end(data_in->v0e[0], data_in->v0e[1], data_in->v0e[2]);
    VectorMax3d edge0_vertex1_end(data_in->v1e[0], data_in->v1e[1], data_in->v1e[2]);
    VectorMax3d edge1_vertex0_end(data_in->v2e[0], data_in->v2e[1], data_in->v2e[2]);
    VectorMax3d edge1_vertex1_end(data_in->v3e[0], data_in->v3e[1], data_in->v3e[2]);
    VectorMax3d p000 = edge0_vertex0_start - edge1_vertex0_start,
                p001 = edge0_vertex0_start - edge1_vertex1_start,
                p011 = edge0_vertex1_start - edge1_vertex1_start,
                p010 = edge0_vertex1_start - edge1_vertex0_start;
    VectorMax3d p100 = edge0_vertex0_end - edge1_vertex0_end,
                p101 = edge0_vertex0_end - edge1_vertex1_end,
                p111 = edge0_vertex1_end - edge1_vertex1_end,
                p110 = edge0_vertex1_end - edge1_vertex0_end;
    Scalar dl = 0;
    Scalar edge0_length = 0;
    Scalar edge1_length = 0;

    dl = 3 * max_linf_4(p000, p001, p011, p010, p100, p101, p111, p110);
    edge0_length =
        3 * max_linf_4(p000, p100, p101, p001, p010, p110, p111, p011);
    edge1_length =
        3 * max_linf_4(p000, p100, p110, p010, p001, p101, p111, p011);
    return VectorMax3d(
        tolerance / dl, tolerance / edge0_length, tolerance / edge1_length);
}

__device__ __host__ void get_numerical_error(
    const VectorMax3d *vertices, const int vsize,
    const bool &check_vf,
    const bool using_minimum_separation,
    Scalar *error)
{
    Scalar eefilter;
    Scalar vffilter;
    if (!using_minimum_separation)
    {
#ifdef GPUTI_USE_DOUBLE_PRECISION
        eefilter = 6.217248937900877e-15;
        vffilter = 6.661338147750939e-15;
#else
        eefilter = 3.337861e-06;
        vffilter = 3.576279e-06;
#endif
    }
    else // using minimum separation
    {
#ifdef GPUTI_USE_DOUBLE_PRECISION
        eefilter = 7.105427357601002e-15;
        vffilter = 7.549516567451064e-15;
#else
        eefilter = 3.814698e-06;
        vffilter = 4.053116e-06;
#endif
    }

    Scalar xmax = fabs(vertices[0].v[0]);
    Scalar ymax = fabs(vertices[0].v[1]);
    Scalar zmax = fabs(vertices[0].v[2]);
    for (int i = 0; i < vsize; i++)
    {
        if (xmax < fabs(vertices[i].v[0]))
        {
            xmax = fabs(vertices[i].v[0]);
        }
        if (ymax < fabs(vertices[i].v[1]))
        {
            ymax = fabs(vertices[i].v[1]);
        }
        if (zmax < fabs(vertices[i].v[2]))
        {
            zmax = fabs(vertices[i].v[2]);
        }
    }
    Scalar delta_x = xmax > 1 ? xmax : 1;
    Scalar delta_y = ymax > 1 ? ymax : 1;
    Scalar delta_z = zmax > 1 ? zmax : 1;
    if (!check_vf)
    {
        error[0] = delta_x * delta_x * delta_x * eefilter;
        error[1] = delta_y * delta_y * delta_y * eefilter;
        error[2] = delta_z * delta_z * delta_z * eefilter;
    }
    else
    {
        error[0] = delta_x * delta_x * delta_x * vffilter;
        error[1] = delta_y * delta_y * delta_y * vffilter;
        error[2] = delta_z * delta_z * delta_z * vffilter;
    }
    return;
}

__device__ bool CCD_Solver(
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
    // return true;
    overflow_flag = 0;
    Scalar *v0s = (data_in->v0s);
    Scalar *v1s = (data_in->v1s);
    Scalar *v2s = (data_in->v2s);
    Scalar *v3s = (data_in->v3s);
    Scalar *v0e = (data_in->v0e);
    Scalar *v1e = (data_in->v1e);
    Scalar *v2e = (data_in->v2e);
    Scalar *v3e = (data_in->v3e);
    
    const int MAX_NO_ZERO_TOI_ITER = SCALAR_LIMIT;
    // unsigned so can be larger than MAX_NO_ZERO_TOI_ITER
    unsigned int no_zero_toi_iter = 0;
    bool is_impacting, tmp_is_impacting;
    Scalar tolerance_in = tolerance;
    Scalar ms_in = ms;
    Scalar *tol = new Scalar[3];
    // this is the error of the whole mesh
    Scalar *err1 = new Scalar[3];
    do
    {
        VectorMax3d tol_v = is_vf ? compute_face_vertex_tolerance_3d_new(data_in, tolerance_in) : compute_edge_edge_tolerance_new(data_in, tolerance_in);
        
        tol[0] = tol_v.v[0];
        tol[1] = tol_v.v[1];
        tol[2] = tol_v.v[2];
        //////////////////////////////////////////////////////////
        
        if (err[0] < 0)
        { // if error[0]<0, means we need to calculate error here
            VectorMax3d *vlist = new VectorMax3d[8];
            for (int i = 0; i < 3; i++)
            {
                vlist[0].v[i] = v0s[i];
                vlist[1].v[i] = v1s[i];
                vlist[2].v[i] = v2s[i];
                vlist[3].v[i] = v3s[i];
                vlist[4].v[i] = v0e[i];
                vlist[5].v[i] = v1e[i];
                vlist[6].v[i] = v2e[i];
                vlist[7].v[i] = v3e[i];
            }

            bool use_ms = ms > 0;
            get_numerical_error(vlist, 8, is_vf, use_ms, err1);
            delete[] vlist;
        }
        else
        {
            err1[0] = err[0];
            err1[1] = err[1];
            err1[2] = err[2];
        }
        //////////////////////////////////////////////////////////
        
        tmp_is_impacting = interval_root_finder_double_horizontal_tree(
            tol, tolerance_in, toi, is_vf, err1, ms_in, v0s,
            v1s, v2s, v3s,
            v0e, v1e, v2e,
            v3e, t_max, max_itr, output_tolerance, overflow_flag);
        
        
        if (overflow_flag)
        {
            delete[] tol;
            delete[] err1;
            return true;
        }
        if (no_zero_toi_iter == 0)
        {
            // This will be the final output because we might need to
            // perform CCD again if the toi is zero. In which case we will
            // use a smaller t_max for more time resolution.
            is_impacting = tmp_is_impacting;
        }
        else
        {
            toi = tmp_is_impacting ? toi : t_max;
        }

        // This modification is for CCD-filtered line-search (e.g., IPC)
        // strategies for dealing with toi = 0:
        // 1. shrink t_max (when reaches max_itr),
        // 2. shrink tolerance (when not reach max_itr and tolerance is big) or
        // ms (when tolerance is too small comparing with ms)
        if (tmp_is_impacting && toi == 0 && no_zero_toi)
        {

            // meaning reaches max_itr, need to shrink the t_max to return a more accurate result to reach target tolerance.
            if (output_tolerance > tolerance_in)
            {
                t_max *= 0.9;
            }
            else
            { // meaning the given tolerance or ms is too large. need to shrink them,
                if (10 * tolerance_in < ms_in)
                { // ms is too large, shrink it
                    ms_in *= 0.5;
                }
                else
                { // tolerance is too large, shrink it
                    tolerance_in *= 0.5;
                }
            }
        }

        no_zero_toi_iter++;

        // Only perform a second iteration if toi == 0.
        // WARNING: This option assumes the initial distance is not zero.
    } while (no_zero_toi && no_zero_toi_iter < MAX_NO_ZERO_TOI_ITER && tmp_is_impacting && toi == 0);
    delete[] err1;
    delete[] tol;
    return is_impacting;
}

__device__ bool vertexFaceCCD_double(
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
    
    bool res = CCD_Solver(data_in, err, ms, toi, tolerance, t_max, max_itr, output_tolerance, no_zero_toi, overflow_flag, true);
    return res;
}

__device__ bool edgeEdgeCCD_double(
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
    bool res = CCD_Solver(data_in, err, ms, toi, tolerance, t_max, max_itr, output_tolerance, no_zero_toi, overflow_flag, false);
    return res;
}