// Engram row hashing, a BigInt port of modeling_embla.py's: token ids -> table row indices (16 a token, in 4 pages of 56 rows).
// Matches the PyTorch reference exactly. The WGSL decoder does the same on the GPU (engramGatherSrc).
const M64=(1n<<64n)-1n, u=x=>x&M64;
const K1=0x9E3779B97F4A7C15n,K2=0xBF58476D1CE4E5B9n,K3=0x94D049BB133111EBn,SEED=0x6A09E667F3BCC909n,SALT=0xD6E8FEB86659FD93n;
const fin=h=>{h=u((h^(h>>30n))*K2);h=u((h^(h>>27n))*K3);return h^(h>>31n)};
const gcd=(a,b)=>b?gcd(b,a%b):a;
export function pageRows(ids,{orders,heads,pageHeads,pages,S,pad}){
  const U=[];for(let s=1;s<S;s++)if(gcd(s,S)===1)U.push(BigInt(s));
  const T=ids.length,maxO=Math.max(...orders),out=[];
  for(let t=0;t<T;t++){
    const lag=L=>(t-L>=0?BigInt(ids[t-L]):BigInt(pad));
    const H={};
    for(let n=1;n<=maxO;n++){let r=SEED;for(let L=n-1;L>=0;L--){r=u((r^lag(L))*K1);r^=r>>29n;}
      H[n]=fin(r^u(0x243F6A8885A308D3n*BigInt(n+1)));}
    const row=[];
    orders.forEach((n,k)=>{for(let g=0;g<heads/pageHeads;g++){
      const hp=fin(H[n]^u(K1*BigInt(1+64*n+g)));
      const page=(hp>>1n)%BigInt(pages), hsl=fin(hp^SALT), a=(hsl>>1n)%BigInt(S), s=U[Number((hsl>>40n)%BigInt(U.length))];
      for(let j=0;j<pageHeads;j++) row.push(Number((BigInt(k)*BigInt(pages)+page)*BigInt(S)+(a+BigInt(j)*s)%BigInt(S)));
    }});
    out.push(row);
  }
  return out;
}
