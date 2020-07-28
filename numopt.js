// This is a Javascript port of scipy's implementation of Powell's conjugate direction method,
// based on the file optimize.py by Travis E. Oliphant

var numopt = (function(){

	function powell(func, x0, opts) {
		'use strict'
		// Find the vector x[n] that minimizes the scalar function f(x), beginning
		// the search at x0[n], which must be an array of floats. Uses Powell's conjugate
		// direction method with Brent as the 1D helper.
		// options:
		//   xtol: tolerance used in the 1d brent search
		//   ftol: tolerance used in the powell search itself
		//   maxiter: max number of powell iterations
		//   maxcall: max number of function calls. Only checked once per powell iteration.
		// returns {x, fval, niter, ncall, status}
		// status is "success" on success, otherwise the fit was a failure.
		opts = Object.assign({xtol: 1e-4, ftol: 1e-4, maxiter: 0, maxcall: 0}, opts);
		var x = x0.slice();
		var n = x.length;
		// Set up iteration bound defaults
		if(!opts.maxiter && !opts.maxfun) opts.maxiter = opts.maxcall = n*1000;
		else if(!opts.maxfun)  opts.maxfun  = Number.POSITIVE_INFINITY;
		else if(!opts.maxiter) opts.maxiter = Number.POSITIVE_INFINITY;
		// Count iterations and evaluations
		var niter = 0, ncall = 0;
		// Set up our initial search vectors
		var dirs = [];
		for(var i = 0; i < n; i++) {
			var dir = new Array(n).fill(0); dir[i] = 1;
			dirs.push(dir);
		}
		var x1   = x.slice();
		var fval = func(x1);
		while(true) {
			var fx = fval;
			var bigind = 0;
			var delta  = 0;
			for(var i = 0; i < n; i++) {
				var dir1 = dirs[i];
				var fx2  = fval;
				var [x, fval, dir1, calls] = linesearch_powell(func, x, dir1, {tol:opts.xtol*100, fval:fval});
				ncall += calls;
				if(fx2 - fval > delta) {
					delta  = fx2 - fval;
					bigind = i;
				}
			}
			niter++;
			// Should we stop iterating?
			if((fx-fval)*2 <= opts.ftol*(Math.abs(fx)+Math.abs(fval))+1e-20) break; // Minimum found
			if(ncall >= opts.maxfun)  break;
			if(niter >= opts.maxiter) break;
			if(!isFinite(fx) || !isFinite(fval)) break;
			// Construct the extrapolated point
			dir1   = x-x1;
			var x2 = 2*x-1;
			x1     = x.slice();
			fx2    = func(x2);

			if(fx > fx2) {
				var t = 2*(fx + fx2 - 2*fval) * (fx-fval-delta)**2 - delta*(fx-fx2)**2;
				if(t < 0) {
					[fval, x, dir1, calls] = linesearch_powell(func, x, dir1, {tol:opts.xtol*100, fval:fval});
					ncall += calls;
					if(any_nonzero(dir1)) {
						dirs[bigind] = dirs[dirs.length-1];
						dirs[dirs.length-1] = dir1;
					}
				}
			}
		}

		var status;
		if     (ncall >= opts.maxfun)  status = "maxcall";
		else if(niter >= opts.maxiter) status = "maxiter";
		else if(!isFinite(fval) || any_invalid(x)) status = "nan";
		else status = "success";

		return {x: x, fval: fval, niter: niter, ncall: ncall, status: status};
	}

	function brent(func, opts) {
		'use strict'
		// Find the minimum of the 1d function func(x).
		// options:
		//   tol:     tolerance of search
		//   maxiter: max number of iterations
		//   xa, xb, xc: these control the area the search is done in. If only xa and xb are
		//    passed, then these will be sent as argument to bracket(). Otherwise, they
		//    should be compatible with the output of bracket()
		// returns {x, fval, niter, ncall}
		var opts  = Object.assign({tol:1.48e-8, maxiter:500, xa:null, xb:null, xc:null}, opts);
		var cg = 0.3819660, mintol = 1e-11;
		// Bracket the minimum if we haven't already given it
		if(opts.xa != null && opts.xb != null && opts.xc != null)
			var [xa, xb, xc, ncall] = [opts.xa, opts.xb, opts.xc, 0];
		else {
			if(opts.xa != null && opts.xb != null)
				var binfo = bracket(func, {fa:opts.fa, fb:opts.fb});
			else
				var binfo = bracket(func);
			var [xa, xb, xc, ncall] = [binfo.xa, binfo.xb, binfo.xc, binfo.ncall];
		}
		var x, w, v, fw, fv, fx;
		x = w = v = xb;
		fw = fv = fx = func(x); ncall++;
		if(xa < xc) var [a,b] = [xa,xc];
		else        var [a,b] = [xc,xa];
		var deltax = 0;
		var niter  = 0;
		while(niter < opts.maxiter) {
			var tol1 = opts.tol * Math.abs(x) + mintol;
			var tol2 = tol1*2;
			var xmid = (a+b)/2;
			// Check for convergence
			if(Math.abs(x-xmid) < tol2-(b-a)/2) break;
			if(Math.abs(deltax) <= tol1) {
				if(x >= xmid) deltax = a-x;
				else          deltax = b-x;
				var rat = cg*deltax;
			} else {
				var tmp1 = (x-w)*(fx-fv);
				var tmp2 = (x-v)*(fx-fw);
				var p    = (x-v)*tmp2 - (x-w)*tmp1;
				tmp2  = 2*(tmp2-tmp1);
				if(tmp2 > 0) p = -p;
				tmp2  = Math.abs(tmp2);
				var dx_temp = deltax;
				deltax      = rat;
				// Check parabolic fit
				if(p > tmp2*(a-x) && p < tmp2*(b-x) && Math.abs(p) < Math.abs(0.5*tmp2*dx_temp)) {
					// Ok, do a parabolic step
					rat = p/tmp2;
					var u = x+rat;
					if(u-a < tol2 || b-u < tol2) {
						if(xmid - x >= 0) rat =  tol1;
						else              rat = -tol1;
					}
				} else {
					// Just do a golden step
					if(x >= xmid) deltax = a-x;
					else          deltax = b-x;
					rat = cg*deltax;
				}
			}
			// Update by at least tol1
			var step = Math.abs(rat) >= tol1 ? rat : rat >= 0 ? tol1 : -tol1;
			u = x + step;
			var fu = func(u); ncall++;

			if(fu > fx) {
				if(u < x) a = u;
				else      b = u;
				if(fu <= fw || w == x) { v = w; w = u; fv = fw; fw = fu; }
				else if(fu <= fv || v == x || v == w) { v = u; fv = fu; }
			} else {
				if( u >= x) a = x;
				else        b = x;
				v  = w;  w  = x;  x  = u;
				fv = fw; fw = fx; fx = fu;
			}
			
			niter++;
		}
		return {x:x, fval:fx, niter:niter, ncall:ncall};
	}

	function bracket(func, opts) {
		'use strict'
		// Bracket the minimum of the 1d function f(x)
		// options:
		//   xa, xb:     starting points for the search. Default to xa = 0, xb = 1
		//   grow_limit: maximum grow limit. Detaults to 110
		//   maxiter:    maximum number of iterations
		// Returns {xa, xb, xc, fa, fb, fc, status}
		opts = Object.assign({grow_limit: 110, maxiter: 1000, xa:0, xb:1}, opts);
		var gold = 1.618034;
		var tiny = 1e-21;
		var xa = opts.xa, fa = func(xa);
		var xb = opts.xb, fb = func(xb);
		if(fa < fb) { [xa,xb] = [xb,xa]; [fa,fb] = [fb,fa]; }
		var xc = xb + gold*(xb-xa), fc = func(xc);
		var ncall = 3, niter = 0;
		while(fc < fb) {
			var tmp1 = (xb-xa)*(fb-fc);
			var tmp2 = (xb-xc)*(fb-fa);
			var val  = tmp2-tmp1;
			var denom = Math.abs(val) < tiny ? 2*tiny : 2*val;
			var w = xb - ((xb-xc)*tmp2 - (xb-xa)*tmp1)/denom;
			var wlim = xb + opts.grow_limit*(xc-xb);
			if(niter > opts.maxiter) return {xa:xa, xb:xb, xc:xc, fa:fa, fb:fb, fc:fc, niter:niter, ncall:ncall, status:"niter"};
			niter++;
			if((w-xc)*(xb-w) > 0) {
				var fw = func(w); ncall++;
				if(fw < fc)
					return {xa:xb, xb:w, xc:xc, fa:fb, fb:fw, fc:fc, niter:niter, ncall:ncall, status:"success"};
				else if(fw > fb) {
					return {xa:xa, xb:xb, xc:w, fa:fa, fb:fb, fc:fw, niter:niter, ncall:ncall, status:"success"};
				} else {
					w  = xc + gold*(xc-xb);
					fw = func(w); ncall++;
				}
			} else if((w-wlim)*(wlim-xc) >= 0) {
				w  = wlim; 
				fw = func(w); ncall++;
			} else if((w-wlim)*(xc-w) > 0) {
				var fw = func(w); ncall++;
				if(fw < fc) {
					xb = xc; xc = w;
					fb = fc; fc = fw;
					w  = xc +  gold*(xc-xb);
					fw = func(w); ncall++;
				}
			} else {
				w  = xc + gold*(xc-xb);
				var fw = func(w); ncall++;
			}
			xa = xb; xb = xc; xc = w;
			fa = fb; fb = fc, fc = fw;
		}
		return {xa:xa, xb:xb, xc:xc, fa:fa, fb:fb, fc:fc, niter:niter, ncall:ncall, status:"success"};
	}

	// Helpers

	function any_nonzero(vec) {
		for(var i = 0; i < vec.length; i++)
			if(vec[i] !=0) return true;
		return false;
	}
	function any_invalid(vec) {
		for(var i = 0; i < vec.length; i++)
			if(!isFinite(vec[i])) return true;
		return false;
	}
	function linesearch_powell(func, x, dir, opts) {
		'use strict'
		opts = Object.assign({tol: 1e-3, fval:null}, opts);
		var n = x.length;
		// Translate from the 1d parameter space given by a to the Nd parameter
		// space func expects.
		function func_1d(a) {
			var x2 = new Array(n);
			for(var i = 0; i < x.length; i++)
				x2[i] = x[i] + a*dir[i];
			return func(x2);
		}
		// Handle the case where the dir is all zero, in which case we can't do anything
		if(!any_nonzero(dir)) {
			var fval = opts.fval, calls = 0;
			if(fval == null) { fval = func(x); calls = 1; }
			return [x, fval, dir, calls];
		} else {
			var res  = brent(func_1d, {tol:opts.tol});
			var x2   = x.slice();
			var dir2 = dir.slice();
			for(var i = 0; i < x.length; i++) {
				x2[i] = x[i] + dir[i]*res.x;
				dir2[i] = dir[i]*res.x;
			}
			return [x2, res.fval, dir2, res.ncall];
		}
	}

	return {
		powell: powell,
		brent: brent,
		bracket: bracket,
	}
})();
