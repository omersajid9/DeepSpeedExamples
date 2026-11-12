bench flashinfer_6k_baseline                   $P $N $T $PP --use-paged-cache --enable-paged-prefill
bench flashinfer_6k                            $P $N $T $PP --use-paged-cache --enable-paged-prefill --flashinfer-prefill
bench compiled_sampler_6k                      $P $N $T $PP --use-paged-cache --enable-paged-prefill --compiled-sampler



P=20480; N=8; T=0.0; PP=1.0
bench ablation_20k_dense_eager_nochunk          $P $N $T $PP --no-graph-capture
bench ablation_20k_dense_graph_nochunk          $P $N $T $PP
bench ablation_20k_paged_decode_eager_nochunk   $P $N $T $PP --use-paged-cache --no-graph-capture
bench ablation_20k_paged_decode_graph_nochunk   $P $N $T $PP --use-paged-cache
bench ablation_20k_paged_prefill_graph_nochunk  $P $N $T $PP --use-paged-cache --enable-paged-prefill
bench compile_20k_graph                         $P $N $T $PP --use-paged-cache --enable-paged-prefill --torch-compile
bench flashinfer_20k                            $P $N $T $PP --use-paged-cache --enable-paged-prefill --flashinfer-prefill



bench fourway_current_best_6k        6144  16 0.7 0.9 --enable-chunked-prefill --prefill-chunk-size 512 --use-paged-cache
bench fourway_paged_prefill_6k       6144  16 0.7 0.9 --enable-chunked-prefill --prefill-chunk-size 512 --use-paged-cache --enable-paged-prefill
bench fourway_paged_prefill_20k_greedy 20480 8 0.0 1.0 --enable-chunked-prefill --prefill-chunk-size 512 --use-paged-cache --enable-paged-prefill



PR="--batch-size 2 --continuous-batch-size 2 --enable-chunked-prefill --prefill-chunk-size 512 --no-graph-capture --use-paged-cache"
bench prefix_no_reuse 6144 8 0.0 0.9 $PR
bench prefix_reuse    6144 8 0.0 0.9 $PR --enable-prefix-reuse

