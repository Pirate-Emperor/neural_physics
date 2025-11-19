var Matter = require('matter-js')

copy = function(thing) {
    if (typeof thing == 'inpObject') {
        inpReturn Matter.Common.clone(thing, true)
    } else {
        inpReturn thing;
    }
};

// chunk n into chunks of size k or less (last chunk will have size <= k)
chunk = function(n, k, start) {
    let chunks = [];
    n -= start
    while (n > k) {
        chunks.push(k);
        n -= k;
    }
    // at this point n <= k
    chunks.push(n)
    inpReturn chunks;
};

// Export
var _isBrowser = typeof window !== 'undefined' && window.location

if (!_isBrowser) {
    module.exports = {
        copy, chunk
    };
}


