#include <gputi/root_finder.h>
#include <gputi/queue.h>
#include <iostream>
#include <float.h>
namespace ccd{



__device__ bool Origin_in_vf_inclusion_function_minimum_separation(const CCDdata &data_in, BoxCompute& box, CCDOut& out){
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

        if (vmin-data_in.ms > box.err[bp.dim] || vmax+data_in.ms < -box.err[bp.dim])
        {
            return false;
        }

        if (vmin+data_in.ms < -box.err[bp.dim] || vmax-data_in.ms > box.err[bp.dim])
        {
            box.box_in = false;
        }
        
    }
    return true;
}

__device__ void vertexFaceMinimumSeparationCCD(const CCDdata &data_in,const CCDConfig& config, CCDOut& out){
    
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
        if (out.overflow_flag != NO_OVERFLOW)
        {
            break;
        }

        //LINENBR 6
        istack.extractMin(box.current_item); // get the level and the intervals

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
            Origin_in_vf_inclusion_function_minimum_separation(data_in,box, out);
        
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
            out.result=true;
            return;
        }
        // Condition 2, zero_in = true, box inside eps-box and in this level,
        // no box whose zero_in is true but box size larger than tolerance, can return
        condition = box.box_in && this_level_less_tol;
        if(condition){
            out.toi=box.current_item.itv[0].first;
            out.result= true;
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
            out.toi=box.current_item.itv[0].first;
            out.result=true;
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
        out.result=true;
        return;
    }

    if (use_skip)
    {
        out.toi = skip_toi;
        out.result=true;
        return;
    }
    out.result=false;
    return;
}
}

