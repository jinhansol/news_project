export default function Loader({ image, text }) {
  return (
    <div className="loader">
      <img src={image} alt="로딩 이미지" className="loader-img" />
      <p>{text || "검색 중입니다..."}</p>
    </div>
  );
}
