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

    // LINENBR 2
    int refine = 0;
    bool zero_in;
    bool condition;
    // level < tolerance. only true, we can return when we find one overlaps eps box and smaller than tolerance or eps-box

    while (!istack.empty())
    {
        //LINENBR 6
        istack.extractMin(box.current_item); // get the level and the intervals
        // LINENBR 8
        refine++;
        zero_in =
            Origin_in_vf_inclusion_function_minimum_separation(data_in,box, out);
        
        if (!zero_in)
            continue;

        // get the width of the box
        box.widths[0] = box.current_item.itv[0].second - box.current_item.itv[0].first;
        box.widths[1] = box.current_item.itv[1].second - box.current_item.itv[1].first;
        box.widths[2] = box.current_item.itv[2].second - box.current_item.itv[2].first;
                
        // LINENBR 15, 16
        // Condition 1, if the tolerance is smaller than the threadshold, return true;
        condition = box.widths[0] <= out.tol[0] && box.widths[1] <= out.tol[1] && box.widths[2] <= out.tol[2];
        if(condition){
            out.result=true;
            return;
        }
        // Condition 2, the box is inside the epsilon box, have a root, return true;
        condition = box.box_in;
        if(condition){
            out.result= true;
            return;
        }

        // Condition 3, real tolerance is smaller than the input tolerance, return true
        condition = box.true_tol <= config.co_domain_tolerance;
        if (condition)
        {
            out.result=true;
            return;
        }

        // LINENBR 12
        if (refine > config.max_itr)
        {
            out.overflow_flag = ITERATION_OVERFLOW;
            out.result=true;
            return;
        }
        split_dimension(out,box);
        bisect_vf_and_push(box,config, istack,out);
        if (out.overflow_flag != NO_OVERFLOW)
        {
            out.result=true;
            return;
        }
    }
    out.result=false;
    return;
}
}

